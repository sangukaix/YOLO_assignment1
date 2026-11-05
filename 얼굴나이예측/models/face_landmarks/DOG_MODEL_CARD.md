---
license: cc-by-nc-4.0
pipeline_tag: keypoint-detection
tags:
  - tflite
  - keypoint-detection
  - facial-landmarks
  - animals
  - dogs
  - mobile
  - dogflw
library_name: tf-keras
---

# Dog Facial Landmarks (DogFLW, 46 points)

![Demo](demo.jpg)

<sub>The 46 landmarks predicted by this repository's models (face localizer, then landmark model) on nine CC0 photos from Wikimedia Commons. Blue: ears. Green: eyes. Orange: nose. Yellow: mouth, chin and forehead. Photos, left to right from the top: [French bulldog](https://commons.wikimedia.org/wiki/File:Bulldog-1047518_1920.jpg) by Yuri; [Karelian Bear Dog](https://commons.wikimedia.org/wiki/File:Karjalankarhukoira,_Karelian_Bear_Dog_looking_at_camera.jpg) by Uusijani; [puppy](https://commons.wikimedia.org/wiki/File:Puppy_filtered.jpg) (Pixabay, photographer not named); [Australian Shepherd](https://commons.wikimedia.org/wiki/File:Australian_Shepherd_in_snow.jpg) (PxHere, photographer not named); [Rottweiler puppy](https://commons.wikimedia.org/wiki/File:Rottweiler_puppy_-21603071920.jpg) by Vlaaitje; [golden retriever](https://commons.wikimedia.org/wiki/File:Dog_with_dumbbell.jpg) by Heidi Sadecky; [puppy](https://commons.wikimedia.org/wiki/File:Puppy_by_Jairo_Alzate,_2015.jpg) by Jairo Alzate; [Jack Russell Terrier](https://commons.wikimedia.org/wiki/File:Black_and_White_Jack_Russell_Terrier.jpg) by United-flags-20; [puppy in Hamburg](https://commons.wikimedia.org/wiki/File:Hamburg_puppy_%28Unsplash%29.jpg) by Eddy Lackmann.</sub>

Two-stage dog face analysis in TFLite: a face localizer that finds the face, and
a landmark detector that predicts the 46-point
[DogFLW](https://github.com/martvelge/DogFLW) scheme on the resulting crop.

These are the models that ship in the
[dog_detection](https://pub.dev/packages/dog_detection) Flutter package. As far
as I can tell they are the first publicly released weights trained on DogFLW.

Training code, the full experiment journal, and the evaluation harness are at
[hugocornellier/dog-face-landmarks-training](https://github.com/hugocornellier/dog-face-landmarks-training).

A Kaggle notebook, [Dog Facial Landmarks on DogFLW
(TFLite)](https://www.kaggle.com/code/hugocornellier/dog-facial-landmarks-on-dogflw-tflite),
runs both stages on a DogFLW image and draws the landmarks.

## Files

| File | Size | What it is |
|---|---|---|
| `dog_face_localizer.tflite` | 16 MB | Stage 1. EfficientNetB2, 224px, predicts one face box |
| `dog_face_landmarks_full.tflite` | 11 MB | Stage 2. MobileNetV3-Large, 384px, 46 landmarks. **The shipped model** |
| `keras/*.keras` | 35 to 62 MB | Full Keras models, for fine-tuning or re-export |
| `metadata/*.json` | | Per-model training config and validation metrics |
| `metadata/*.csv` | | Full per-epoch training logs |

Both `.tflite` files are float16 **static-shape** exports (batch-1 concrete
function). In the landmark model, the ReLU after each deconv is also a separate
op instead of being fused into `TRANSPOSE_CONV`. The GPU delegate needs both to
run the whole graph, and `scripts/reexport_static.py` in the training repository
does both. On an M4 Max the landmark stage runs 27.10 ms on XNNPACK CPU against
3.82 ms on GPU via CompiledModel.

## Accuracy

NME_IOD (normalized mean error, inter-ocular distance) on the DogFLW validation
split of 480 images. Lower is better.

| Model | Backbone | Res | NME_IOD | Size |
|---|---|---|---|---|
| `dog_face_landmarks_full` | MobileNetV3-Large | 384 | **8.56** | 11 MB |

Until 24 September 2026 this repository also had a larger EfficientNetV2-S
landmark model (8.77, 55 MB). The 11 MB model beats it, so it was withdrawn. It
is still in the repository history.

Best result in the project is 8.04, from a 3-model EfficientNetV2-S ensemble at
256+320+384 with multi-scale and flip TTA, at 18 forward passes. Those weights
are not published here; the journal has the recipe.

For reference, the DogFLW paper's ELD ensemble reaches 6.52. This project started
near 40.

Localizer: 0.79 bbox IoU. Trained on 3,853 images, validated on 480.

**Dogs are much harder than cats.** The same architecture reaches 3.48 on CatFLW
and 8.56 here. Ear landmarks carry the error at NME 12 to 14 against roughly 5
for eyes, ear tips reach 15 to 18, and the train-val gap sits near 3.4 across
every model tried and did not respond to regularization. The journal works
through why.

## Input and output contract

**Localizer** takes `[1, 224, 224, 3]` float32 in `[0, 1]`, letterboxed to
square. It returns `bbox_xyxy`, normalized `[0, 1]` in letterboxed coordinates.
Undo the letterbox to get image coordinates.

**Landmarks** takes `[1, 384, 384, 3]` float32 in `[0, 1]`: crop the image to the
face box expanded by a **0.1 margin**, then resize to square. It returns
`landmarks_xy` of shape `[1, 92]`, flattened `[x0, y0, x1, y1, ... x45, y45]`,
normalized `[0, 1]` **relative to the crop**, not the original image. Map them
back through the same crop to get image coordinates.

Rescaling to `[0, 255]` for the backbone happens inside the graph. Do not do it
yourself.

Exact per-model config, including every augmentation setting, is in
`metadata/*.json`.

## Usage

Python:

```python
# pip install ai-edge-litert
import os
import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.cpu_options import CpuOptions
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import Options

# Set the thread count: left at its default, the CPU ran single-threaded in my tests.
options = Options(HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=os.cpu_count()))
model = CompiledModel.from_file("dog_face_landmarks_full.tflite", options=options)
inputs, outputs = model.create_input_buffers(0), model.create_output_buffers(0)

# crop: face box + 0.1 margin, resized to 384x384, float32 in [0, 1]
inputs[0].write(crop[None].astype(np.float32))
model.run_by_index(0, inputs, outputs)
xy = outputs[0].read(92, np.float32).reshape(46, 2)  # normalized to the crop
```

For the GPU, pass `HardwareAccelerator.GPU | HardwareAccelerator.CPU` instead:
on an M4 Max the landmark model then runs entirely on the GPU, in about 2 ms
from Python. CompiledModel logs some INFO lines and an "NPU accelerator could
not be loaded" warning to stderr when it loads a model; both are harmless.
LiteRT's `Interpreter` (`ai_edge_litert.interpreter`) runs these files too.

The [demo
notebook](https://www.kaggle.com/code/hugocornellier/dog-facial-landmarks-on-dogflw-tflite)
runs both stages, localizer included, on a DogFLW image.

Flutter: use [dog_detection](https://pub.dev/packages/dog_detection), which wires
both stages together, handles the crop math, and adds a species gate.

## License

**CC BY-NC 4.0. Non-commercial use only.** See `LICENSE`.

These weights are derived from the DogFLW dataset, which is CC BY-NC 4.0. I asked
the dataset authors directly how they wanted weights trained on their annotations
to be licensed. They asked for CC BY-NC 4.0 rather than a permissive license, to
stay consistent with the non-commercial terms of the source data, and granted
permission to publish on that basis.

If you need commercial use, that permission is not mine alone to give. Contact
the dataset authors at the Tech4Animals Lab, University of Haifa.

The **training code** in the GitHub repository is Apache 2.0. Only the weights
are non-commercial.

## Citation

Please cite the DogFLW paper:

```bibtex
@article{martvel2025dog,
  title={Dog facial landmarks detection and its applications for facial analysis},
  author={Martvel, George and Zamansky, Anna and Pedretti, Giulia and Canori,
          Chiara and Shimshoni, Ilan and Bremhorst, Annika},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={21886},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## Acknowledgements

Thanks to George Martvel, Anna Zamansky, Giulia Pedretti, Chiara Canori, Ilan
Shimshoni, and Annika Bremhorst at the Tech4Animals Lab, University of Haifa, for
publishing DogFLW and for permission to release these weights.
