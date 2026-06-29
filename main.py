
import tensorflow_datasets as tfds
import matplotlib.pyplot as  plt
import numpy as np

def generate_dataset():


    ds_train, ds_test = tfds.load(
        'rock_paper_scissors',
        split=['train', 'test'],
        as_supervised=True
    )


    return ds_train, ds_test



    


def main():
    
    print("---main---")

if __name__ == "__main__":
    main()
    