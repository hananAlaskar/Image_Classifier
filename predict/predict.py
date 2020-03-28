
from get_predict_input_args import get_predict_input_args
from load_checkpoint import load_checkpoint
from predict_image import predict
from view_classify import view_classify


def main():

    in_arg = get_predict_input_args()

    checkpoint = in_arg.checkpoint
    model = load_checkpoint(checkpoint)

    image_path = in_arg.image_path
    top_k = in_arg.top_k
    probs, classes_indexs = predict(image_path, model, top_k)

    file_name = in_arg.category_names
    view_classify(probs[0].numpy(), classes_indexs[0].numpy(), model, file_name)


if __name__ == "__main__":
    main()