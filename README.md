# pyg-face_landmark
GNN implementation for Face Landmark detection


Run: 
python main_pyg.py --gnn gcn --batch_size 20 --epochs 2 --rootdir <path_to_root_dir> --dataset <dataset_name> --filedir <path_to_output_saving_dir> --metric roc-auc-approx-0.5 --emb_dim 400 --num_layer 3  --num_post_fnn 2


Note:
Available evaluation metric are:
- mse 
- rmse 
- r2 
- roc-auc-naive 
- roc-auc-approx-<--ratio-->

Add a float (>0.0 and <1.0) in place of the <--ratio--> placeholder. 

Here we use generalized roc-auc score in our regression task.
roc-auc-naive metric is O(n^2) complex. 
So we introduced roc-auc-approx metric which will approximate the roc-auc-naive.
closer the provided ratio to 1.0, better the approximation. (default ratio is set to 0.5).

User raw images in testing/ directory for testing purposes.
This raw dataset contains 10 RGB images.
