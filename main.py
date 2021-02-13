import argparse
import learning
import tensorflow.python.keras
from keras import layers, optimizers, Sequential, metrics, losses


def build_model():
    model = Sequential()
    model.add(layers.Dense(30, activation='relu', input_shape=[30]))

    model.add(layers.Dropout(0.8))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.8))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.F1])
    model.summary()
    return model


def getArguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''IoV-MD''')
    parser.add_argument('-at', '--attack_type',
                        help='dataset choice (svc,rf,lr)',
                        choices=['DoSRandom', 'RandomPos', 'ConstPos', 'DoS', 'DoSRandomSybil', 'DoSDisruptive',
                                 'RandomPosOffset', 'Disruptive', 'RandomSpeed', 'ConstPosOffset', 'DelayedMessages',
                                 'ConstSpeed', 'DataReplay', 'RandomSpeedOffset', 'TrafficSybil', 'EventualStop',
                                 'DoSDisruptiveSybil', 'ConstSpeedOffset', 'DataReplaySybil'],
                        required=True)
    parser.add_argument('-t', '--time',
                        help='time choice(0709/1416)',
                        choices=['0709', '1416'],
                        required=True)
    parser.add_argument('-k', '--kfold',
                        help='Kfold number (1-5).',
                        type=int,
                        required=True)

    return parser.parse_args()


def main(argus):
    model = build_model()
    kfold = argus.kfold
    data_path = 'VeReMi-Dataset/' + argus.at + '_' + argus.t
    learning.learn(model, data_path)


if __name__ == "__main__":
    args = getArguments()
    main(args)
