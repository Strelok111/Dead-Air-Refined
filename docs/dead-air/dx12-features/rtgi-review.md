# Multi-bounce RTGI: implementation audit and acceptance criteria

Status: **NOT IMPLEMENTED / NOT VERIFIED**. This document audits the unfinished `feature/rtgi-multibounce` branch. The original indirect illumination remains active.

## Findings confirmed from code

1. `src/Layers/xrRenderPC_R4/da_rt_passes.cpp`: `da_rt_gi_pass` is currently a stub. There is no `rt_gi.hlsl`, GI temporal shader or GI filter pipeline. `RtEffect::GI` is never marked ready. Do not present this as functional RTGI.
2. `src/Layers/xrRenderPC_R4/r4_rendertarget_phase_rt.cpp`: the GI buffers are only placeholders. Their allocation is now gated by actual GI pass readiness. `r__rtgi` defaults to zero until the passes are complete.
3. `src/Layers/xrRenderPC_R4/da_rt.cpp`: the deferred-combine GI switch must check a **successful GI dispatch for the current frame** (`gi_frame == Device.dwFrame`), not just the requested effect setting. A dispatch failure must preserve the old indirect term, never use a stale/black buffer.
4. `packaging/dead-air-x64/compatibility/gamedata/shaders/r3/combine_1.ps`: its `hdiffuse` term is the legacy irradiance/ambient diffuse contribution. The current guard can select a GI texture later, but does not replace every indirect-lighting path in the game (the optional `r2_gi` reflected-light contribution and forward/HUD/vegetation/particle lighting need separate treatment).
5. `src/Layers/xrRender_R2/r2_R_lights.cpp` and `r2_R_render.cpp`: `render_lights` consumes/clears light lists before `phase_combine`. A GI dispatch **cannot** read `RImplementation.Lights.package` in the combine phase to gather active local lights. Snapshot lights earlier, and retain owner-point-light identity rather than six `OMNIPART` shadow faces.
6. `packaging/dead-air-x64/compatibility/gamedata/shaders/r3/rt/rt_material.hlsli`: `rt_hit_light` includes an ambient/hemisphere approximation. Do **not** inject it at every bounce: doing so double counts ambient energy. Split incident direct lighting, sky miss radiance, and emissive radiance; transport with a properly normalized BRDF and PDF.
7. `src/Layers/xrRenderPC_R4/da_rt_scene.cpp`: static meshes, tree stand-ins, near dynamic rigid meshes, and a bounded number of skinned actors are represented; several procedural plants, particles, fluids and transparent elements are not. This limits the correctness of "entire game" indirect GI until proxies/material handling are improved.
8. RT is only supported by the DX12/Vulkan RT lanes with MSAA disabled (`da_rt_available`). DirectX 11 and unsupported hardware must preserve the existing indirect-lighting fallback.

## Target transport equation

For the indirect diffuse component at a primary surface `x0`, with Lambertian albedo `rho`, choose each secondary direction using the cosine-hemisphere PDF `p(w) = cos(theta)/pi`, so the path throughput multiplier is `rho`. Accumulate direct illumination at each **secondary** vertex with shadow-tested light sampling, and the environment on a ray miss:

`L_GI(x0) = E[ sum(k = 1..B) ( product(j = 0..k-1) rho_j ) L_direct(x_k) + (product(j=0..m-1) rho_j) L_sky(w_m) ]`

This expression is schematic; the actual estimator must include the full BSDF/PDF ratios when sampling other than Lambertian cosine-weighted directions, emitter-sampling PDFs, Russian-roulette probabilities, and light-selection PDFs. Avoid counting primary direct light again in the GI result. For point/spot lights, do not omit their contributions simply because they are missing from the original `RtShade`.

## Required rendering integration

1. **Before local lights are consumed:** capture a bounded or tiled/clustered set of active point/spot lights, their world position/direction, radiance, radius, cutoff/spot cone, and light/projector information into a GPU-visible buffer, with stable ids for temporal invalidation. Preserve six-face point-light ownership.
2. **Scene preparation:** use static/dynamic TLAS; augment missing geometries with suitable proxies or document omissions. Respect alpha testing, two-sided materials, emissive intensity, shading normals and consistent linear-color transformations.
3. **Trace:** reconstruct primary world position/normal from the G-buffer; trace one or more cosine-weighted diffuse directions, then recursively continue over 2–4+ bounces depending on quality, with self-intersection offsets and bounded path length. At secondary vertices evaluate direct sun + scene lights with visibility rays; at misses evaluate sky; add emissive hits. Consider Russian roulette after later bounces.
4. **Denoise:** trace at reduced resolution when needed, but use full-resolution G-buffer-guided reconstruction. Keep previous radiance and at least depth/normal validation; reject history on geometry/material/light or camera discontinuities. Use variance clamping and edge-aware à-trous passes, or a validated equivalent.
5. **Combine:** use GI as the opaque diffuse indirect irradiance input; **retain direct lighting** from sun/local lights, and existing specular/reflection paths until explicitly improved. Avoid applying an AO scalar to physically shadow-tested GI a second time.
6. **Additional materials:** investigate static precomputed hemisphere visibility, optional `r2_gi` reflected lights, dynamic/forward translucency, HUD/character lighting, foliage, water and emissive particles. Suppress legacy indirect only where the RTGI solution actually replaces it.
7. **Fault tolerance:** leave RTGI off when any shader/pipeline/target is missing. On frame failures, avoid stale GI substitution and record diagnostics. Expose a toggle only when working, with a debug mode for raw, temporal, filtered, and final GI.

## Validation required before claiming completion

- Compile DX12 **and Vulkan** shader variants with their actual DXC/Slang/SPIR-V toolchains, and Windows engine configurations including non-unity. Compile the DirectX 11 fallback.
- Verify the RTGI path uses the correct bindless resources, target/UAV transitions, non-aliased SRV/UAV passes, and sync barriers between dispatches.
- Compare indoor rooms, open terrain, covered areas, nighttime scenes, moving torches/campfires, color-bleeding walls, skinned actors, vegetation and camera cuts. Use reference path tracing or controlled irradiance tests where possible.
- Validate that toggling GI cannot create stale/black frames; examine high-frequency light changes, disocclusion, ghosting, noise, fireflies and negative radiance.
- Benchmark GPU and VRAM at 1080p, 1440p and 4K, and test Low/Medium/High quality budgets. Distinguish tracing, temporal, denoising and TLAS costs with GPU timestamps.

**No DX12/Vulkan runtime test or successful CI build has been verified for this feature branch.** Its current GPU tracing passes do not exist, and a functioning image cannot be claimed.
