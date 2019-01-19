This project is forked from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.

I modified only some codes in Cityscapes labels->photos evaluation part so that we can use our own dataset. Since the ground truth labels are not available in the Cityscapes dataset provided by [Junyanz](https://affinelayer.com/pixsrv/), I downloaded them from the Cityscapes dataset website, and renamed them to match the file names of street scene images in testA from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets.

To use the new evaluation codes, just run the following command:\\
python ./scripts/eval_cityscapes/evaluate.py --label_dir /path_to_labels --result_dir /path_to_generated_photos --output_dir /path_to_store_results

