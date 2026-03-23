## Plan: Multi-Pipeline Temporal Canopy Delineation

This approach splits the workflow into two independent pipelines to handle the resolution disparity (1cm RGB vs. 5cm MS) and temporal frequency without massive data resampling. We extract ultra-precise geometries from the 1cm RGB using DINO+SAM2/3, robust phenological boundaries from the 5cm MS using Detectree2 across all 10 dates, and merge them into a single definitive canopy map using a spatial consensus aggregator. 

**Note on Data:** The 1cm RGB and 5cm MS data are coregistered and captured simultaneously.

**Phase 1: High-Resolution Baseline (RGB Pipeline)**
1. Configure and run a single-image pipeline on the 1cm P1 RGB data to generate base geometries.
2. Use the `tilerizer` (set to 1cm/px), followed by the DINO Swin-L `detector`.
3. Clear internal bounding-box duplicate detections with an `aggregator` (`iou` algorithm).
4. Prompt the `sam3` `segmenter` to generate the highly detailed spatial masks from the boxes.
5. Create a new custom `edt_splitter` component. This calculates the Euclidean Distance Transform (EDT) and runs a Watershed algorithm on the SAM3 masks (optionally heavily weighted by the high-res DSM canopy peaks) to cleanly slice any large fused crowns into individual tree instances.

**Phase 2: Temporal Phenology (MS Pipeline)**
1. Fine-tune the `detectree2` segmenter on the 5cm Multispectral data using your 6,000 ground truth polygons to recognize species-specific phenological signatures. (Detectree2 natively supports `IMGMODE: ms` to handle >3 channels).
2. Configure a second pipeline using only the `tilerizer` (set to 5cm/px) and the fine-tuned `detectree2` `segmenter`.
3. Loop over all 10 dates (5 dates/year * 2 years) per site independently and output 10 GeoPackages of multispectral canopy masks per site.

**Phase 3: Multi-Sensor Temporal Aggregation & Delineation Refinement**
1. Build a custom standalone script (`TemporalAggregator`) to ingest the 1 high-res RGB GeoPackage and the 10 coregistered MS temporal GeoPackages. 
2. **Temporal Fission (Splitting Trees):** Analyze polygon overlaps across time. If the RGB pipeline incorrectly fused two adjacent trees into a single large polygon, but the MS pipeline consistently detects two separate crowns across multiple temporal dates (e.g., due to asynchronous phenology like one blooming earlier), use the MS masks to forcefully split the base RGB geometry.
3. **Temporal Fusion (Refining Outlines):** Use the temporal stack to dynamically refine the tree boundaries. Compute the union mask across the multitemporal MS layers (capturing the tree at peak leaf-on), and use this to "inflate" or smooth the baseline RGB boundary where the 1cm data may have under-segmented shadowed branches. 
4. Implement a "Temporal Voting" threshold: e.g., if an RGB tree geometry overlaps with a Detectree2 MS polygon in at least 4 out of 10 timestamps, confirm the tree and assign the extracted multispectral temporal curves to the polygon attributes. 

**Relevant files**
- `canopyrs/engine/pipeline.py` — orchestrates the two distinct standard pipeline runs.
- `canopyrs/engine/models/segmenter/detectree2.py` — handles the multi-channel MS input organically. 
- `canopyrs/engine/components/aggregator.py` — reference for building the standalone multi-vector fusion script.
- `canopyrs/tools/segmentation/train_detectree2.py` — (assuming analogous script) used for fine-tuning Detectree2 on the overlapping MS data.

**Verification**
1. Run both pipelines on a single 1-hectare test plot. Because the orthomosaics are already perfectly coregistered, we can skip complex geometric alignment and directly validate how tightly the 5cm MS polygons snap to the 1cm RGB geometries.
2. Verify the Detectree2 configuration successfully loaded 5 channels by inspecting the component initialization logs.
3. Evaluate the final fused polygons against the 6,000 ground truth polygons using IoU (Intersection over Union) to prove the "Temporal Voting" filter removed false positives.

**Decisions**
- Kept the 1cm RGB and 5cm MS separate to prevent massive resampling data bloat.
- Treat each temporal MS date independently to leverage standard Mask R-CNN (Detectree2) architecture without altering the neural network's graph to accept 50 continuous channels.
- Ground truth polygons will be rasterized specifically for the Detectree2 MS fine-tuning logic.
