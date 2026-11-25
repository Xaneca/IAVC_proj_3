## Dados

### Age and Gender:
- UTKFace (Aligned and Cropped): https://www.kaggle.com/datasets/moritzm00/utkface-cropped?resource=download
- License:

The labels of each face image is embedded in the file name, formated like 
[age]_[gender]_[race]_[date&time].jpg

### Face Detection:
- WIDER Face (test): http://shuoyang1213.me/WIDERFACE/
- License:

### Expression
- AffectNet: https://www.kaggle.com/datasets/yakhyokhuja/affectnetaligned
- License: 

## PreProcess

### UTKFace

- Discretization in 10 year bins, except for children where we separated into 5 year bins:
    - [0-4], [5-9], [10-19], [20-30], ... [80+]
- Train/Validation/Test: 70/15/15 - stratified per age and gender
- docs/figures have data statistics in distribution
- Images resized from 200x200 to 112x112 (no detection/alignment) -> data/processed/utkface_112
- In the future: resolve wrongly labeled data
    - image of an eye not a person, adults mislabeeled as children, ...