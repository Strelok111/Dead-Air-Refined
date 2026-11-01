// Two-pass depth/normal/colour-guided a-trous filter for RGB indirect irradiance.
#include "rt_common.hlsli"
#include "da_material_codes.h"
cbuffer rt_gi_filter_constants : register(b0)
{
    RtView view;
    float4 filter; // x step, y normal exponent, z depth-plane tolerance, w colour scale
};
Texture2D<float4> gi_source : register(t0);
Texture2D<float> gi_depth : register(t1);
Texture2D<float4> gi_position : register(t2);
Texture2D<float4> gi_normal : register(t3);
RWTexture2D<float4> gi_out : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(view.size.xy)))
        return;
    const float d = gi_depth.Load(int3(id.xy, 0));
    const float4 centre = gi_source.Load(int3(id.xy, 0));
    if (rt_is_sky(d))
    {
        gi_out[id.xy] = float4(centre.rgb, 1.0f);
        return;
    }
    const float z = rt_linear_z(d, view);
    const RtSurface base = rt_surface(gi_position, gi_normal, int2(id.xy), view);
    const float material_word = gi_position.Load(int3(id.xy, 0)).w;
    const float material_code = (view.scene.z & RT_VIEW_GBUFFER_PACKED) ?
        gbuf_unpack_mtl(material_word) : abs(material_word);
    if (da_is_foliage(material_code))
    {
        gi_out[id.xy] = 0.0f;
        return;
    }
    const float3 p = rt_view_pos(float2(id.xy) + 0.5f, z, view);
    const float kernel[3] = {0.375f, 0.25f, 0.0625f};
    const int step = int(filter.x);
    const float scale = 0.04f + filter.z * z;
    const float centre_l = dot(centre.rgb, float3(0.2126f, 0.7152f, 0.0722f));
    const float colour_scale = 0.15f + filter.w * sqrt(max(centre_l, 0.0f));
    float3 sum = 0.0f;
    float sum_w = 0.0f;
    [unroll] for (int y = -2; y <= 2; ++y)
    {
        [unroll] for (int x = -2; x <= 2; ++x)
        {
            const int2 at = int2(id.xy) + int2(x, y) * step;
            if (any(at < 0) || any(at >= int2(view.size.xy)))
                continue;
            const float dep = gi_depth.Load(int3(at, 0));
            if (rt_is_sky(dep))
                continue;
            const float qz = rt_linear_z(dep, view);
            const RtSurface neighbour = rt_surface(gi_position, gi_normal, at, view);
            const float mw = gi_position.Load(int3(at, 0)).w;
            const float code = (view.scene.z & RT_VIEW_GBUFFER_PACKED) ? gbuf_unpack_mtl(mw) : abs(mw);
            if (da_is_foliage(code))
                continue;
            const float3 q = rt_view_pos(float2(at) + 0.5f, qz, view);
            const float plane = abs(dot(base.n_view, q - p));
            const float4 v = gi_source.Load(int3(at, 0));
            const float l = dot(v.rgb, float3(0.2126f, 0.7152f, 0.0722f));
            const float weight = kernel[abs(x)] * kernel[abs(y)] *
                exp(-plane / scale) * pow(saturate(dot(base.n_view, neighbour.n_view)), filter.y) *
                exp(-abs(l - centre_l) / colour_scale);
            sum += v.rgb * weight;
            sum_w += weight;
        }
    }
    gi_out[id.xy] = float4(sum_w > 1e-6f ? sum / sum_w : centre.rgb, 1.0f);
}