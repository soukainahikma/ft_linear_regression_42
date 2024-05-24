
import pickle
import sys


class Prediction:

    def __init__(self) -> None:
        self.teta0 = 0
        self.teta1 = 0
        self.price = 0
        self.mileage = 0

    def get_input(self):
        try:
            mileage = float(input("Enter mileage value in km: "))
            if mileage < 0:
                raise Exception
            self.mileage = mileage
        except Exception:
            sys.exit('You have entered a negative value')

    def get_file_info(self, path):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
                self.teta1, self.teta0 = model
        except Exception as error:
            sys.exit(error)

    def estimate(self):
        self.price = self.teta0 + self.teta1 * self.mileage
        print(f'The estimated price is : {self.price}')


if __name__ == '__main__':
    test = Prediction()
    test.get_file_info('model.pkl')
    test.get_input()
    test.estimate()
