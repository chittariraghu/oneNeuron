def main():

from utils.model import Perceptron
from utils.all_utils import prepare_data
from utils.all_utils import save_model
from utils.all_utils import save_plot
#from utils.all_utils import _plot_decision_regions
import pandas as pd
import numpy as numpy



OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1],
}

df = pd.DataFrame(OR)

df


X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()
save_model(model,filename="or.model")
save_plot(df,"or.png",model)

if __name__=='__main__':
    main()