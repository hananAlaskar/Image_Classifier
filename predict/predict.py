
from get_predict_input_args import get_predict_input_args
from load_checkpoint import load_checkpoint
from predict_image import predict


def main():

    in_arg = get_predict_input_args()

    checkpoint = in_arg.checkpoint
    model = load_checkpoint(checkpoint)

    image_path = in_arg.image_path
    top_k = in_arg.top_k
    probs, classes_indexs = predict(image_path, model, top_k)


if __name__ == "__main__":
    main()