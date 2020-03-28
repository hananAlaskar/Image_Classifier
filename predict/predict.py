
from get_predict_input_args import get_predict_input_args
from load_checkpoint import load_checkpoint


def main():

    in_arg = get_predict_input_args()

    checkpoint = in_arg.checkpoint
    model = load_checkpoint(checkpoint)



if __name__ == "__main__":
    main()