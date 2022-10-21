
This code requires Pytorch Geometric version>=2.0.2 and torch version>=1.10.1.


Run following cmd:
	python main_pyg.py --gnn <GNN_TYPE> --filename <FILENAME>

Set GNN_TYPE to one of the following:
	gcn
	gin

Set FILENAME to the filepath to which results are saved.

py2graph.py is a script that converts a Python code snippet into a graph object that is fully compatible with the OGB code dataset.
Here I put code to convert images into graphs in the same way done right now.