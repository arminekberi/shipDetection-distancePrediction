import os

out = 'yolo_dataset_v4'
abs_out = os.path.abspath(out)
with open(os.path.join(out, 'dataset.yaml'), 'w') as f:
    f.write(f'path: {abs_out}\n')
    f.write('train: images/train\n')
    f.write('val: images/val\n')
    f.write('names:\n  0: boat\n')
print(f'wrote {out}/dataset.yaml')
