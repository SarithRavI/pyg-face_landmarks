from torchmetrics import R2Score,MeanSquaredError
import torch

class Evaluator:
    def __init__(self,type="MSE",**kwargs):
        self.type = type
        self.kwargs = kwargs
        if self.type == "MSE":
            if "squared" in self.kwargs:
                self.evaluator = MeanSquaredError(squared = self.kwargs["squared"])
            else:
                self.evaluator = MeanSquaredError()

    def eval(self,input_dict):
        
        landmarks_ref_list = input_dict["landmark_ref"]
        landmarks_pred_list = input_dict["landmark_pred"]
        num_landmarks = self.kwargs["num_landmarks"]

        if self.type == "R2" and self.evaluator is None:
            if "multioutput" in self.kwargs.keys():
                self.evaluator = R2Score(num_landmarks,self.kwargs["multioutput"])
        else:
            self.evaluator = R2Score(num_landmarks)
        metric = 0
        for i,pred in enumerate(landmarks_pred_list):
            output = torch.tensor(pred)
            target = torch.tensor(landmarks_ref_list[i])
            metric += self.evaluator(output,target).item()
        return metric


