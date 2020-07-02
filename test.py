from model.activations import Hyper, Hypo
import torch

def main():
    print('hello world!')

    l = [i*.1 for i in range(-6, 6)]
    t = torch.Tensor(l)
    print(f'data: {t}')

    hyper = Hyper(offset=.3)
    print(f'hyper(data): {hyper(t)}')

    hypo = Hypo(min=0, max=.3)
    print(f'hypo(data): {hypo(t)}')


if __name__ == "__main__":
    main()
