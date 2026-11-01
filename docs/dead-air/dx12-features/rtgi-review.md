# DX12 multi-bounce RTGI: experimental implementation and acceptance criteria

Status: **Experimental source implementation; Windows engine build, DXC shader compilation and runtime QA are NOT verified.**
Scope: `xrRender_R5.dll` **DirectX 12 only** (DXR 1.1 inline ray queries and HLSL 2021 at cs_6_5).
The Vulkan renderer is unchanged and this feature is inactive on it; DirectX 11 retains its existing lighting.

## Effect and control

`r__rtgi 0|1` defaults **off**; change the value and run `vid_restart` so GI pipelines and render targets are created.
GI is independent of the quality presets and does not appear in the Video menu yet.
`r__rt_quality`: Low 2 bounces, half resolution; Medium 3 bounces, half resolution; High 4 bounces, full resolution.
`r__rt_debug 6` displays GI irradiance when a GI frame was produced.
`r__gpu_log` prints `rt_gi` for the tracing + reconstruction + filtering passes.

With successful GI the opaque deferred combine substitutes denoised GI irradiance for the `hmodel.h`
hemisphere/ambient diffuse contribution. Existing raster sun and local direct illumination, material BRDF specular,
and ray-traced/screen-space reflections remain separate. The legacy local reflected-light sprites (`R2FLAG_GI`)
are suppressed *only on a frame with a successfully produced GI result*. On failure or unsupported hardware
the original indirect paths remain. HUD pixels retain their old ambient because the HUD uses a different projection.

Vegetation and particles deliberately retain the old lighting and are not traced by GI.
Trunks, foliage crowns and procedural plants are excluded from GI's ray mask;
static geometry, ordinary alpha-tested geometry, supported dynamic/rigid and nearby skinned models are included.
Transparent water/glass and procedural effect geometry are not represented as full physical path-tracing materials.

## Source components

- `src/Layers/xrRenderPC_R4/da_rt_passes.cpp`: DX12 GI pipeline creation, capture of two active light packages,
  ray-tracing dispatch, full-resolution history reconstruction, two RGB a-trous passes.
- `r2_R_render.cpp`: snapshot and dispatch before the two local light packages are consumed;
  this is deliberately earlier than `phase_combine`, which cannot see the consumed light lists.
- `r2_R_lights.cpp`: suppress the legacy reflected-light sprites for successful GI frames, not for failed ones.
- `r4_rendertarget_phase_rt.cpp`: GI raw, temporal RGB/frame-length histories, world-normal/depth histories,
  filter intermediate, final irradiance; allocation is gated by GI pipeline readiness.
- `rt/rt_gi.hlsl`: cosine-weighted multi-bounce Lambertian transport with explicit direct-light sampling
  on secondary hits, sun and local-light visibility rays, emissive hits, sky misses and Russian roulette.
  The primary albedo is intentionally excluded from the path throughput and is applied by `combine_1.ps`.
- `rt/rt_gi_temporal.hlsl`: full-resolution depth-aware reconstruction, previous world-normal/depth history,
  reprojection using motion vectors where available, luminance-space clipping.
- `rt/rt_gi_filter.hlsl`: two depth/normal/colour-guided a-trous steps.
- `combine_1.ps`: the final GI result replaces the opaque diffuse indirect term only for a valid pixel
  (alpha > 0.5); vegetation and unsupported HUD/sky pixels keep their earlier path.

## Limitations to review before enabling by default

1. **Not compiled or rendered yet**. Shader source and C++ code have been integrated without a Windows build.
   Native DXC, root descriptor validation, resource transitions, and frame rendering must be checked on the game.
2. **Light budget**: 48 selected point/spot emitters at most from both packages (deduplicating
   `OMNIPART` faces to their owning point light). Importance uses camera distance and intensity.
   On scenes with more relevant emitters, indirect lighting may omit some; selection should ideally use
   per-tile or per-bounce light lists, not one global camera-priority list.
3. **Nonphysical legacy materials**: secondary texture samples, emissive masks and weather/sky cubemaps
   follow existing engine conventions; full energy-calibrated linear albedo, Fresnel and GGX path transport
   are not yet implemented. This is *diffuse GI*, not full path-traced specular/refraction.
4. **No explicit light-change history invalidation**: moving light changes may ghost until temporally refreshed.
   The history does reject world-normal/depth mismatch, device reset and camera/history resets.
5. **Incomplete representations**: no traced vegetation, particles, transparent liquids or full procedural snow;
   mesh overlays or novel mod materials must be tested. Primary unsupported pixels preserve the old indirect term.
6. **High settings may be expensive**: four bounces at native output resolution, shadow visibility queries,
   and up to 48 local-light candidate checks per hit. Measure GPU timings and VRAM.
7. **Testing gaps**: verify startup with only GI selected, GI+shadows+AO+reflections, all settings off,
   shader-compile failure, console toggles and restarts, weather transitions, emissive and flashlight movement,
   overcast/nighttime outdoors, level transitions, save compatibility, and fallback on unsupported DX12 adapters.

## Validation procedure

Use the supported CMake/Ninja build workflow (`PROJECT_RULES.md`), not hand-written project files:
`tools\build\build_x64.ps1 -Configuration Release -Clean`; repeat incremental build until no work remains.
Check the DX12 layer tests and the native shader pipeline compile reports. Start the game with `-r5`
and `-dxdebug` and inspect `! [rt]`, DXC and D3D12 debug-layer errors.

Test a simple enclosed white/coloured room, exterior courtyard, tunnels, moving sun,
a torch moving behind a corner, multiple lamps, skinned characters and `qa_pedro`/lamp-hall saves.
Capture GI debug 6 and lighting with `r__rtgi 0/1`, identical weather/time/angle and shader cache.
Verify actual reflected colour, correct indirect shadowing, no double ambient/legacy reflected lights,
disocclusion, normal changes, no temporal ghosting or black frames, and unchanged vegetation.
Measure `rt_gi` GPU time and compare GPU frame ms, frame p95/p99, VRAM and level load.

Do not label the feature production-ready or merge to `main` without a successful DX12 compiler,
full game runtime and visual QA pass. This branch is for code review and iteration.
