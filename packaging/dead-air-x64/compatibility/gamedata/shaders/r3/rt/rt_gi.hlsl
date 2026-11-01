// Multi-bounce diffuse indirect illumination, DX12/DXR 1.1 (cs_6_5).
// The primary albedo is multiplied by combine_1, NOT here. The cosine-sampled
// secondary throughput is a product of the hit albedos only.
#include "rt_material.hlsli"
#include "da_material_codes.h"

#define GI_LIGHTS 48
#define GI_MASK (RT_MASK_STATIC | RT_MASK_STATIC_ALPHA | RT_MASK_DYNAMIC | RT_MASK_SKINNED | RT_MASK_NOSHADOW)

cbuffer rt_gi_constants : register(b0)
{
    RtView view;
    RtShade shade;
    float4 gi_params; // xy trace size, z bounce limit, w maximum path distance (metres)
    uint4 gi_count;  // x selected local lights
    float4 gi_position[GI_LIGHTS];  // xyz world position, w = reciprocal squared range
    float4 gi_colour[GI_LIGHTS];    // rgb intensity, w = spotlight half-cone cosine (-2 for point)
    float4 gi_direction[GI_LIGHTS]; // xyz direction from spotlight
    float4 gi_falloff; // share, soft-core hardness, gain
};
Texture2D<float> gi_depth : register(t0);
Texture2D<float4> gi_position_buffer : register(t1);
Texture2D<float4> gi_normal_buffer : register(t2);
TextureCube gi_sky0 : register(t3);
TextureCube gi_sky1 : register(t4);
RWTexture2D<float4> gi_raw : register(u0);

// A small independently scrambled sequence for each pixel, frame and bounce.
float gi_random(inout uint seed)
{
    seed ^= seed >> 16;
    seed *= 0x7feb352du;
    seed ^= seed >> 15;
    seed *= 0x846ca68bu;
    seed ^= seed >> 16;
    return float(seed & 0x00ffffffu) * (1.0f / 16777216.0f);
}

struct GiHit
{
    float t;
    uint instance_id, instance_index, geometry_index, primitive;
    float2 bary;
    float3x4 o2w;
};

GiHit gi_trace(RayDesc ray, float cone)
{
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(rt_scene, RAY_FLAG_NONE, GI_MASK, ray);
    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            rt_alpha_solid(query.CandidateInstanceID(), query.CandidateInstanceIndex(),
                query.CandidateGeometryIndex(), query.CandidatePrimitiveIndex(),
                query.CandidateTriangleBarycentrics(), query.CandidateObjectToWorld3x4(),
                cone + 0.001f * query.CandidateTriangleRayT(), view))
            query.CommitNonOpaqueTriangleHit();
    }
    GiHit hit = (GiHit)0;
    hit.t = -1.0f;
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        hit.t = query.CommittedRayT();
        hit.instance_id = query.CommittedInstanceID();
        hit.instance_index = query.CommittedInstanceIndex();
        hit.geometry_index = query.CommittedGeometryIndex();
        hit.primitive = query.CommittedPrimitiveIndex();
        hit.bary = query.CommittedTriangleBarycentrics();
        hit.o2w = query.CommittedObjectToWorld3x4();
    }
    return hit;
}

bool gi_visible(float3 p, float3 n, float3 dir, float distance, float cone)
{
    if (distance <= 0.04f)
        return true;
    RayDesc ray;
    ray.Origin = p + n * 0.018f + dir * 0.006f;
    ray.Direction = dir;
    ray.TMin = 0.012f;
    ray.TMax = distance;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(rt_scene, RAY_FLAG_NONE, GI_MASK, ray);
    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            rt_alpha_solid(query.CandidateInstanceID(), query.CandidateInstanceIndex(),
                query.CandidateGeometryIndex(), query.CandidatePrimitiveIndex(),
                query.CandidateTriangleBarycentrics(), query.CandidateObjectToWorld3x4(),
                cone, view))
            query.CommitNonOpaqueTriangleHit();
    }
    return query.CommittedStatus() != COMMITTED_TRIANGLE_HIT;
}

float gi_light_attenuation(uint i, float3 point, out float3 direction, out float max_distance)
{
    const float4 light = gi_position[i];
    const float3 delta = light.xyz - point;
    const float dist2 = dot(delta, delta);
    const float x2 = dist2 * light.w;
    max_distance = sqrt(max(dist2, 1e-6f));
    direction = delta / max(max_distance, 1e-4f);
    if (x2 >= 1.0f)
        return 0.0f;
    // Matches the renderer's default soft-core inverse-square lamp falloff:
    // da_light_falloff.h with share=1, core=4 and gain=1.7.
    const float window = saturate(1.0f - x2 * x2);
    const float stock = saturate(1.0f - x2);
    const float shaped = gi_falloff.z * window * window / (1.0f + gi_falloff.y * x2);
    float attenuation = lerp(stock * stock, shaped, saturate(gi_falloff.x));
    if (gi_colour[i].w > -1.5f)
    {
        const float cone = gi_colour[i].w;
        attenuation *= smoothstep(cone, lerp(cone, 1.0f, 0.3f),
            dot(-direction, gi_direction[i].xyz));
    }
    return attenuation;
}

// Direct lighting at a SECONDARY hit only. The primary direct lights are already in the accumulator.
// Sample one eligible local light using a CDF, and divide its contribution by its selection PDF.
float3 gi_direct(float3 p, float3 n, inout uint seed)
{
    float3 E = 0.0f;
    const float ndotl = saturate(dot(n, view.sun.xyz));
    if (ndotl > 0.0f && gi_visible(p, n, view.sun.xyz, 1000.0f, 0.004f))
        E += max(shade.sun_color.rgb, 0.0f) * ndotl;

    const uint count = min(gi_count.x, (uint)GI_LIGHTS);
    float total = 0.0f;
    [loop] for (uint i = 0u; i < count; ++i)
    {
        float3 d;
        float distance;
        const float att = gi_light_attenuation(i, p, d, distance);
        total += att * saturate(dot(n, d)) *
            dot(max(gi_colour[i].rgb, 0.0f), float3(0.2126f, 0.7152f, 0.0722f));
    }
    if (total < 1e-6f)
        return E;

    const float ticket = gi_random(seed) * total;
    float running = 0.0f;
    [loop] for (uint i = 0u; i < count; ++i)
    {
        float3 d;
        float distance;
        const float att = gi_light_attenuation(i, p, d, distance);
        const float ndl = saturate(dot(n, d));
        const float importance = att * ndl *
            dot(max(gi_colour[i].rgb, 0.0f), float3(0.2126f, 0.7152f, 0.0722f));
        running += importance;
        if (importance > 0.0f && (running >= ticket || i + 1u == count))
        {
            if (gi_visible(p, n, d, max(distance - 0.035f, 0.0f), 0.003f))
                E += max(gi_colour[i].rgb, 0.0f) * (att * ndl * total / importance);
            break;
        }
    }
    return E;
}

float3 gi_sky(float3 direction)
{
    return max(shade.sky_tint.rgb, 0.0f) *
        lerp(gi_sky0.SampleLevel(rt_linear_clamp, direction, 0).rgb,
             gi_sky1.SampleLevel(rt_linear_clamp, direction, 0).rgb,
             saturate(shade.env_color.w)) * 0.66f;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(gi_params.xy)))
        return;
    uint2 pixel = id.xy;
    float depth = gi_depth.Load(int3(pixel, 0));
    if (gi_params.x < view.size.x)
    {
        pixel = min(id.xy * 2u, uint2(view.size.xy) - 1u);
        depth = gi_depth.Load(int3(pixel, 0));
        [unroll] for (uint j = 1u; j < 4u; ++j)
        {
            const uint2 candidate = min(id.xy * 2u + uint2(j & 1u, j >> 1u), uint2(view.size.xy) - 1u);
            const float d = gi_depth.Load(int3(candidate, 0));
            if (d < depth)
            {
                pixel = candidate;
                depth = d;
            }
        }
    }

    if (rt_is_sky(depth))
    {
        gi_raw[id.xy] = 0.0f;
        return;
    }
    const float z = rt_linear_z(depth, view);
    const RtSurface surface = rt_surface(gi_position_buffer, gi_normal_buffer, int2(pixel), view);
    if (rt_is_hud(z, surface))
    {
        gi_raw[id.xy] = 0.0f;
        return;
    }
    const float3 origin = rt_to_world(rt_view_pos(float2(pixel) + 0.5f, z, view), view);
    float3 normal = normalize(rt_dir_to_world(surface.n_view, view));
    float3 point = origin;
    float3 throughput = 1.0f;
    float3 indirect = 0.0f;
    float cone = rt_pixel_size(z, view);
    uint seed = pixel.x * 1973u ^ pixel.y * 9277u ^ view.scene.w * 26699u ^ 0x68bc21ebu;
    const uint bounces = clamp((uint)gi_params.z, 1u, 5u);

    [loop] for (uint bounce = 0u; bounce < bounces; ++bounce)
    {
        float2 random_dir = float2(gi_random(seed), gi_random(seed));
        const float3 direction = rt_cosine_dir(normal, random_dir);
        RayDesc ray;
        ray.Origin = point + normal * 0.02f + direction * 0.004f;
        ray.Direction = direction;
        ray.TMin = 0.015f;
        ray.TMax = gi_params.w;
        const GiHit hit = gi_trace(ray, cone);
        if (hit.t < 0.0f)
        {
            indirect += throughput * gi_sky(direction);
            break;
        }
        point = ray.Origin + ray.Direction * hit.t;
        const RtGeometry geometry = rt_geometry_of(hit.instance_id, hit.geometry_index);
        const RtInstance instance = rt_instance_of(hit.instance_index, view);
        const RtTriangle triangle = rt_fetch(geometry, hit.primitive);
        const RtHitSurface surface_hit = rt_hit_surface(geometry, instance, triangle, hit.bary, hit.o2w,
            direction, cone + 0.001f * hit.t, view);
        normal = normalize(surface_hit.normal);
        const float3 albedo = saturate(surface_hit.albedo);
        if (surface_hit.emissive)
            indirect += throughput * albedo;
        // For cosine-hemisphere sampling, f*cos(theta)/pdf = albedo.
        throughput *= albedo;
        indirect += throughput * gi_direct(point, normal, seed);
        cone += 0.002f * hit.t;

        // Russian roulette after the second bounce: divide by survival probability.
        if (bounce >= 2u)
        {
            const float survive = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.1f, 0.95f);
            if (gi_random(seed) > survive)
                break;
            throughput /= survive;
        }
    }
    gi_raw[id.xy] = float4(min(max(indirect, 0.0f), 32.0f), z);
}