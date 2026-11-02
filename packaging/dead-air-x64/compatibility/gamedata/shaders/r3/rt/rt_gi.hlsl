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
    uint4 gi_count;  // x selected local lights, y first-ray diagnostic, z camera-side primary bias
    float4 gi_position[GI_LIGHTS];  // xyz world position, w = reciprocal squared range
    float4 gi_colour[GI_LIGHTS];    // rgb intensity, w = spotlight half-cone cosine (-2 for point)
    float4 gi_direction[GI_LIGHTS]; // xyz direction from spotlight
    float4 gi_falloff; // share, soft-core hardness, gain
    float4 gi_hud_decompress; // first-person HUD's separate projection
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
    float3x4 o2w, w2o;
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
        hit.w2o = query.CommittedWorldToObject3x4();
    }
    return hit;
}

// Visibility rays originate at a certified surface spawn point, not from a fixed
// centimetre-scale n/dir bias. The light is not a triangle endpoint.
bool gi_visible(float3 spawn, float3 dir, float distance, float cone)
{
    if (distance <= 0.001f)
        return true;
    RayDesc ray;
    ray.Origin = spawn;
    ray.Direction = dir;
    ray.TMin = 0.0f;
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

float gi_light_attenuation(uint i, float3 surface_position, out float3 direction, out float max_distance)
{
    const float4 light = gi_position[i];
    const float3 delta = light.xyz - surface_position;
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
float3 gi_direct(float3 p, float3 spawn, float3 n, float3 geometric_n, inout uint seed)
{
    float3 E = 0.0f;
    const float ndotl = saturate(dot(n, view.sun.xyz));
    if (ndotl > 0.0f && dot(geometric_n, view.sun.xyz) > 0.0f &&
        gi_visible(spawn, view.sun.xyz, 1000.0f, 0.004f))
        E += max(shade.sun_color.rgb, 0.0f) * ndotl;

    const uint count = min(gi_count.x, (uint)GI_LIGHTS);
    float total = 0.0f;
    [loop] for (uint i = 0u; i < count; ++i)
    {
        float3 d;
        float distance;
        const float att = gi_light_attenuation(i, p, d, distance);
        // Use the same geometric-hemisphere eligibility in the CDF and PDF.
        total += (dot(geometric_n, d) > 0.0f ? att * saturate(dot(n, d)) : 0.0f) *
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
        const float importance = (dot(geometric_n, d) > 0.0f ? att * ndl : 0.0f) *
            dot(max(gi_colour[i].rgb, 0.0f), float3(0.2126f, 0.7152f, 0.0722f));
        running += importance;
        if (importance > 0.0f && (running >= ticket || i + 1u == count))
        {
            if (gi_visible(spawn, d, max(distance - 0.001f, 0.0f), 0.003f))
                E += max(gi_colour[i].rgb, 0.0f) * (att * ndl * total / importance);
            break;
        }
    }
    return E;
}

float3 gi_sky(float3 direction)
{
    // The direct-sky tint fades to zero with overcast. GI must still see the cloudy sky:
    // keep the hemisphere/environment colour as its minimum radiance tint.
    const float3 tint = max(shade.sky_tint.rgb, shade.env_color.rgb * 0.5f);
    return max(tint, 0.0f) *
        lerp(gi_sky0.SampleLevel(rt_linear_clamp, direction, 0).rgb,
             gi_sky1.SampleLevel(rt_linear_clamp, direction, 0).rgb,
             saturate(shade.env_color.w)) * 0.66f;
}

// NVIDIA, "Solving Self-Intersection Artifacts in DirectX Raytracing"
// (2023), Listings 1-8. Reconstruct the exact DXR triangle hit in object
// space, include reconstruction / transformation / traversal roundoff, and
// offset along the true geometric normal. No ad hoc hit rejection or cm bias.
// The c1 constant is tuned for NVIDIA RTX; verify on other GPU vendors.
struct GiSpawn
{
    float3 position;
    float3 front;
    float3 normal;
};

bool gi_spawn_from_hit(GiHit hit, RtTriangle tri, float3 incoming, out GiSpawn spawn)
{
    spawn = (GiSpawn)0;
    const float3 v0 = tri.v0.position;
    precise float3 edge1 = tri.v1.position - v0;
    precise float3 edge2 = tri.v2.position - v0;
    const float3 obj_normal = cross(edge1, edge2);
    if (dot(obj_normal, obj_normal) < 1e-20f)
        return false;

    // Add the base vertex last to bound object-space reconstruction error.
    precise float3 obj_position = v0 + mad(hit.bary.x, edge1, hit.bary.y * edge2);

    // DXR's ObjectToWorld3x4 uses column vectors. Add translation last.
    precise float3 world_position;
    world_position.x = hit.o2w._m03 +
        mad(hit.o2w._m00, obj_position.x,
        mad(hit.o2w._m01, obj_position.y, hit.o2w._m02 * obj_position.z));
    world_position.y = hit.o2w._m13 +
        mad(hit.o2w._m10, obj_position.x,
        mad(hit.o2w._m11, obj_position.y, hit.o2w._m12 * obj_position.z));
    world_position.z = hit.o2w._m23 +
        mad(hit.o2w._m20, obj_position.x,
        mad(hit.o2w._m21, obj_position.y, hit.o2w._m22 * obj_position.z));

    // Inverse transpose is required for nonuniform scaling and shear.
    float3 world_normal = mul(transpose((float3x3)hit.w2o), obj_normal);
    const float normal_length_sq = dot(world_normal, world_normal);
    if (normal_length_sq < 1e-20f)
        return false;
    const float world_scale = rsqrt(normal_length_sq);
    world_normal *= world_scale;
    if (dot(world_normal, incoming) > 0.0f)
        world_normal = -world_normal;

    const float c0 = 5.9604644775390625e-8f;
    const float c1 = 1.7881397695873602e-7f;
    const float c2 = 1.1920931797249068e-7f;

    const float3 extent3 = abs(edge1) + abs(edge2) + abs(abs(edge1) - abs(edge2));
    const float extent = max(max(extent3.x, extent3.y), extent3.z);
    float3 obj_error = mad(c0, abs(v0), c1 * extent);
    const float3 world_translation = float3(hit.o2w._m03, hit.o2w._m13, hit.o2w._m23);
    const float3 world_error = mad(c1, mul(abs((float3x3)hit.o2w), abs(obj_position)),
        c2 * abs(world_translation));
    obj_error = mad(c2, mul(abs(hit.w2o), float4(abs(world_position), 1.0f)), obj_error);

    const float offset = dot(obj_error, abs(obj_normal)) * world_scale +
        dot(world_error, abs(world_normal));
    spawn.position = world_position;
    spawn.normal = world_normal;
    spawn.front = mad(offset, world_normal, world_position);
    return true;
}

// G-buffer pixels have neither primitive ID nor barycentrics. Recover them by
// tracing backwards from the camera to the visible pixel's DXR triangle.
// Reject nonmatching RT geometry, rather than misidentifying a nearby prop.
// This incurs one additional inline ray query per non-HUD primary GI sample.
bool gi_primary_spawn(float3 raster_position, float raster_z, float cone, out GiSpawn spawn)
{
    spawn = (GiSpawn)0;
    const float3 from_camera = raster_position - view.camera.xyz;
    const float camera_distance = length(from_camera);
    if (camera_distance < 0.002f)
        return false;

    RayDesc ray;
    ray.Origin = view.camera.xyz;
    ray.Direction = from_camera / camera_distance;
    ray.TMin = 0.0f;
    ray.TMax = camera_distance + max(0.005f, rt_pixel_size(raster_z, view));
    const GiHit hit = gi_trace(ray, cone);
    if (hit.t < 0.0f)
        return false;
    const RtGeometry geometry = rt_geometry_of(hit.instance_id, hit.geometry_index);
    const RtTriangle tri = rt_fetch(geometry, hit.primitive);
    if (!gi_spawn_from_hit(hit, tri, ray.Direction, spawn))
        return false;

    const float hit_z = rt_dir_to_view(spawn.position - view.camera.xyz, view).z;
    if (abs(hit_z - raster_z) > max(0.008f, raster_z * 0.0007f))
        return false;
    return true;
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
    const float material_word = gi_position_buffer.Load(int3(pixel, 0)).w;
    const float material_code = (view.scene.z & RT_VIEW_GBUFFER_PACKED) ?
        gbuf_unpack_mtl(material_word) : abs(material_word);
    if (da_is_foliage(material_code))
    {
        gi_raw[id.xy] = 0.0f;
        return;
    }
    // The first-person weapon and hands are drawn into the G-buffer with a different
    // field of view. Their G-buffer z is correct; the main projection's decoded depth
    // and view ray are not. They receive world-space GI without adding fake HUD
    // meshes to the world TLAS (so they do not cast world-space shadows).
    const bool hud = rt_is_hud(z, surface);
    const float primary_z = hud ? surface.z : z;
    const float4 decompression = hud ? gi_hud_decompress : view.pos_decompress;
    const float2 primary_xy = float2(pixel) + 0.5f;
    const float3 primary_view = float3(primary_z * (primary_xy * decompression.zw - decompression.xy), primary_z);
    const float3 origin = rt_to_world(primary_view, view);
    float3 normal = normalize(rt_dir_to_world(surface.n_view, view));
    float cone = rt_pixel_size(primary_z, view) *
        (hud ? abs(gi_hud_decompress.w / view.pos_decompress.w) : 1.0f);
    uint seed = pixel.x * 1973u ^ pixel.y * 9277u ^ view.scene.w * 26699u ^ 0x68bc21ebu;
    const uint bounces = clamp((uint)gi_params.z, 1u, 5u);
    const float3 toward_camera = normalize(view.camera.xyz - origin);
    GiSpawn primary_spawn = (GiSpawn)0;
    const bool primary_matched = !hud && gi_count.z != 0u &&
        gi_primary_spawn(origin, primary_z, cone, primary_spawn);
    float3 geometric_normal = primary_matched ? primary_spawn.normal : normal;
    float3 bounce_position = origin;
    // HUD geometry and unmatched raster surfaces have no matching DXR triangle.
    // This explicit fallback is NOT the NVIDIA-certified path.
    const float coord = max(max(abs(origin.x), abs(origin.y)), abs(origin.z));
    const float fallback_offset = max(0.001f, coord * 2.5e-7f);
    float3 bounce_front = primary_matched ? primary_spawn.front :
        origin + toward_camera * fallback_offset;
    float3 throughput = 1.0f;
    float3 indirect = 0.0f;

    [loop] for (uint bounce = 0u; bounce < bounces; ++bounce)
    {
        float2 random_dir = float2(gi_random(seed), gi_random(seed));
        // Retain the shading-normal detail where possible. Never emit rays
        // through the true triangle plane (NVIDIA's grazing-angle caveat).
        float3 direction = rt_cosine_dir(normal, random_dir);
        if (dot(direction, geometric_normal) <= 0.0f)
            direction = rt_cosine_dir(geometric_normal, float2(gi_random(seed), gi_random(seed)));
        RayDesc ray;
        const bool first_bounce = bounce == 0u;
        // Mode 0 is a diagnostic A/B of the legacy primary offset.
        ray.Origin = first_bounce && gi_count.z == 0u ?
            origin + normal * 0.002f + direction * 0.0005f : bounce_front;
        ray.Direction = direction;
        ray.TMin = first_bounce && gi_count.z == 0u ? 0.001f : 0.0f;
        ray.TMax = gi_params.w;
        const GiHit hit = gi_trace(ray, cone);
        // First-ray intersection diagnostic, shown by r__rt_debug 10.
        // Red: very close hit (possible self-intersection); green: sky miss;
        // blue: more distant scene hit. Red alone is not proof of a bad hit
        // because legitimate adjacent geometry may also be very close.
        if (first_bounce && gi_count.y != 0u)
        {
            const float near_limit = max(0.04f, 0.0005f * primary_z);
            const float3 classification = hit.t < 0.0f ? float3(0.0f, 1.0f, 0.0f) :
                hit.t < near_limit ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 0.2f, 1.0f);
            gi_raw[id.xy] = float4(classification, primary_z);
            return;
        }
        if (hit.t < 0.0f)
        {
            indirect += throughput * gi_sky(direction);
            break;
        }
        bounce_position = ray.Origin + ray.Direction * hit.t;
        const RtGeometry geometry = rt_geometry_of(hit.instance_id, hit.geometry_index);
        const RtInstance instance = rt_instance_of(hit.instance_index, view);
        const RtTriangle hit_triangle = rt_fetch(geometry, hit.primitive);
        const RtHitSurface surface_hit = rt_hit_surface(geometry, instance, hit_triangle, hit.bary, hit.o2w,
            direction, cone + 0.001f * hit.t, view);
        normal = normalize(surface_hit.normal);
        // Secondary and shadow rays share the exact error-bounded spawn point.
        GiSpawn next_spawn;
        if (gi_spawn_from_hit(hit, hit_triangle, direction, next_spawn))
        {
            bounce_position = next_spawn.position;
            bounce_front = next_spawn.front;
            geometric_normal = next_spawn.normal;
        }
        else
        {
            // Degenerate triangle: conservative fallback only.
            geometric_normal = normal;
            bounce_front = bounce_position + geometric_normal * 0.001f;
        }
        const float3 albedo = saturate(surface_hit.albedo);
        if (surface_hit.emissive)
            indirect += throughput * albedo;
        // For cosine-hemisphere sampling, f*cos(theta)/pdf = albedo.
        throughput *= albedo;
        indirect += throughput * gi_direct(bounce_position, bounce_front, normal, geometric_normal, seed);
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
    gi_raw[id.xy] = float4(min(max(indirect, 0.0f), 32.0f), primary_z);
}