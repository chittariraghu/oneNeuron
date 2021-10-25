from utils.model import Perceptron
from utils.all_utils import prepare_data
from utils.all_utils import save_model
from utils.all_utils import save_plot
#from utils.all_utils import _plot_decision_regions
import pandas as pd
import numpy as numpy

def main(data,eta,epochs,filename,plotFileName):

    df = pd.DataFrame(AND)

    df


    X,y = prepare_data(df)



    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()
    save_model(model,filename=filename)
    save_plot(df,plotFileName,model)
if __name__=='__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
main(data=AND,eta=ETA,epochs=EPOCHS,filename="and.model",plotFileName='and.png')