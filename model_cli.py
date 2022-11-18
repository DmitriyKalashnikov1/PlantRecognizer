from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import sys
import argparse

cultures = {
    0:"подсолнечник",
    1:"картофель",
    2:"пшеница озимая",
    3:"гречиха",
    4: "кукуруза",
    5: "пшеница яровая",
    6: "сахарная свекла"
}


pathToBest = './LSTM_Dense_best96_8%.h5'

def predict_crop(x):
    return np.argmax(model.predict(x))

def createParser():
    parser = argparse.ArgumentParser(
        prog='PlantRecognizer CLI',
        description="This is CLI of the PlantRecognizer AI model, on input it need vector of 69 NVDI indexes, on output it give predicted culture and it index",
        epilog="Author: Dmitriy Kalashnikov, email: slagterra2017@yandex.ru"
    )
    parser.add_argument("-i", '--input', default="std_in", help="Choose input method between std_in and csv file")
    parser.add_argument('-o', '--output', default="std_out", help="Choose output method between std_out and csv file")
    parser.add_argument("-m", "--model", default=pathToBest, help="Choose AI model file")
    return parser

if __name__ == "__main__":
    parser = createParser()
    args = parser.parse_args(sys.argv[1:])
    model = load_model(args.model)
    x = []
    if (args.input == "std_in"):
        for f in range(69):
            x += [float(input(f"Input the {f+1} NDVI index: "))]
            x = np.array(x)
    else:
        df = pd.read_csv(args.input)
        x = df.iloc[0].to_numpy()[0:69]
        #print(x.shape)
        if(len(x) > 69):
            raise ValueError
    y = predict_crop(x.reshape([1,69]))
    if(args.output == 'std_out'):
        print(f"Index: {y}, culture: {cultures[y]}")
    else:
        outputValues = [y, cultures[y]]
        #print(outputValues)
        df = pd.DataFrame([outputValues], columns=["Index", "Culture"])
        df.to_csv(args.output, index=False)


