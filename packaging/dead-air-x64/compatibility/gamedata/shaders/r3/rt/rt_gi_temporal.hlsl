// Full-resolution history reconstruction from half/full-res GI rays, with depth
// and normal rejection, moment clipping, camera/velocity reprojection.
#include "rt_common.hlsli"
cbuffer rt_gi_temporal_constants : register(b0)
{
    RtView view;
    float4 temporal; // x trace width, y trace height, z pixels per trace sample, w minimum fresh share
};
Texture2D<float4> gi_sample : register(t0);
Texture2D<float4> gi_prev_colour : register(t1);
Texture2D<float4> gi_prev_depth : register(t2);
Texture2D<float> gi_depth : register(t3);
Texture2D<float4> gi_position : register(t4);
Texture2D<float4> gi_normal : register(t5);
Texture2D<float2> gi_velocity : register(t6);
RWTexture2D<float4> gi_history_out : register(u0);
RWTexture2D<float4> gi_depth_out : register(u1);

float3 gi_tm(float3 c)
{
    return max(c, 0.0f) / (1.0f + max(c.r, max(c.g, c.b)));
}
float3 gi_untm(float3 c)
{
    return c / max(1.0f - max(c.r, max(c.g, c.b)), 0.025f);
}
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(view.size.xy)))
        return;
    const float d = gi_depth.Load(int3(id.xy, 0));
    if (rt_is_sky(d))
    {
        gi_history_out[id.xy] = 0.0f;
        gi_depth_out[id.xy] = float4(0.0f, 0.0f, 0.0f, view.proj.w);
        return;
    }
    const float z = rt_linear_z(d, view);
    const RtSurface here = rt_surface(gi_position, gi_normal, int2(id.xy), view);
    const float3 n_world = normalize(rt_dir_to_world(here.n_view, view));
    gi_depth_out[id.xy] = float4(n_world, rt_is_hud(z, here) ? here.z : z);
    const bool hud = rt_is_hud(z, here);
    const float primary_z = hud ? here.z : z;

    const int2 texel = int2(id.xy / uint(temporal.z));
    const int2 last = int2(temporal.xy) - 1;
    const float3 primary = rt_view_pos(float2(id.xy) + 0.5f, z, view);
    float3 sum = 0.0f, square = 0.0f;
    float total = 0.0f;

    // Reconstruct across nearby trace blocks, rejecting samples belonging to
    // a different foreground surface. GI is intentionally a colour signal,
    // not the occlusion scalar of rt_temporal.hlsl.
    [unroll] for (int y = -1; y <= 1; ++y)
    {
        [unroll] for (int x = -1; x <= 1; ++x)
        {
            const int2 t = clamp(texel + int2(x, y), int2(0, 0), last);
            const float4 traced_sample = gi_sample.Load(int3(t, 0));
            if (traced_sample.a <= 0.0f)
                continue;
            // The tracer chooses the nearest-depth pixel of each half-resolution block.
            // Check that surface's normal before borrowing its indirect illumination.
            uint2 source_pixel = min(uint2(t) * uint(temporal.z), uint2(view.size.xy) - 1u);
            if (temporal.z > 1.0f)
            {
                float nearest = gi_depth.Load(int3(source_pixel, 0));
                [unroll] for (uint k = 1u; k < 4u; ++k)
                {
                    const uint2 candidate = min(uint2(t) * 2u + uint2(k & 1u, k >> 1u),
                        uint2(view.size.xy) - 1u);
                    const float candidate_depth = gi_depth.Load(int3(candidate, 0));
                    if (candidate_depth < nearest)
                    {
                        nearest = candidate_depth;
                        source_pixel = candidate;
                    }
                }
            }
            const RtSurface source_surface = rt_surface(gi_position, gi_normal, int2(source_pixel), view);
            const float source_z = rt_linear_z(gi_depth.Load(int3(source_pixel, 0)), view);
            if (rt_is_hud(source_z, source_surface) != hud)
                continue;
            const float3 source_normal = normalize(rt_dir_to_world(source_surface.n_view, view));
            if (dot(n_world, source_normal) < 0.85f)
                continue;
            const float2 centre = (float2(t) + 0.5f) * temporal.z;
            const float2 delta = (centre - (float2(id.xy) + 0.5f)) / temporal.z;
            const float dz = abs(traced_sample.a - primary_z);
            const float plane_scale = max(0.04f + 0.02f * z, 0.04f);
            const float w = exp(-0.65f * dot(delta, delta)) * exp(-dz / plane_scale);
            const float3 c = gi_tm(traced_sample.rgb);
            sum += c * w;
            square += c * c * w;
            total += w;
        }
    }
    const float3 current = total > 1e-5f ? sum / total : 0.0f;
    const float3 sigma = total > 1e-5f ? sqrt(max(square / total - current * current, 0.0f)) : 0.0f;
    float3 result = current;
    float frames = total > 1e-5f ? 1.0f : 0.0f;

    // HUD geometry follows a separate animated pose and projection. Its world-camera
    // reprojection cannot be trusted, so accumulate it spatially but not temporally.
    if (!hud && total > 1e-5f && (view.scene.z & RT_VIEW_HISTORY_VALID))
    {
        const float3 world = rt_to_world(primary, view);
        const float3 prev = rt_project(world, view.vp_prev);
        float2 prev_uv = prev.xy;
        if (view.scene.z & RT_VIEW_VELOCITY)
        {
            const float2 motion = gi_velocity.Load(int3(id.xy, 0));
            prev_uv = (float2(id.xy) + 0.5f) * view.size.zw - float2(motion.x, -motion.y) * 0.5f;
        }
        if (prev.z > 0.0f && all(prev_uv > 0.0f) && all(prev_uv < 1.0f))
        {
            const float4 old_surface = gi_prev_depth.SampleLevel(rt_point_clamp, prev_uv, 0);
            if (abs(old_surface.w - prev.z) < 0.04f + 0.02f * prev.z &&
                dot(old_surface.xyz, n_world) > 0.85f)
            {
                const float4 history = gi_prev_colour.SampleLevel(rt_linear_clamp, prev_uv, 0);
                float3 h = gi_tm(history.rgb);
                h = clamp(h, current - 1.75f * sigma - 0.035f, current + 1.75f * sigma + 0.035f);
                frames = min(max(history.a, 1.0f) + 1.0f, 32.0f);
                const float alpha = max(temporal.w, 1.0f / frames);
                result = lerp(h, current, alpha);
            }
        }
    }
    gi_history_out[id.xy] = float4(gi_untm(saturate(result)), frames);
}