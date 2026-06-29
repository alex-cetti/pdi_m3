import numpy as np
import tensorflow as tf
from filtragem import *
from morfologia import apply_limiar, erosao_cv2, dilatacao_cv2


def apply_salt_pepper(image, amount=0.15):
    image = normalize(image)

    mask = tf.random.uniform(tf.shape(image))
    image = tf.where(mask < amount / 2, 0.0, image)
    image = tf.where(mask > 1 - amount / 2, 1.0, image)

    return image


def resize(image, new_size):
    return tf.image.resize(image, [new_size, new_size])


def normalize(image):
    if image.dtype == tf.float32:
        return image
    return tf.cast(image, tf.float32) / 255.0


def to_grayscale(image):
    return tf.image.rgb_to_grayscale(image)

def pipe_filtragem_espacial(image):
    resized = resize(image, 64)
    normalized = normalize(resized)
    gray = to_grayscale(normalized)
    denoised = median_blur(gray)             #remove o sp
    contrasted = apply_gamma(denoised, 1.5)  # gamma > 1: escurece a mao em imagens muito claras
    final_image = sharpening(contrasted)     # sharpening leve (amount=1.0)

    return final_image


def pipe_morfologia(image):
    kernel = np.ones((3, 3), dtype=np.uint8)

    def _apply(img_np):
        img_2d = (img_np[:, :, 0] * 255).astype(np.uint8)
        limiar = int(img_2d.mean())
        img_bin = apply_limiar(img_2d, limiar)

        # abertura = erosao + dilatacao
        img_open = dilatacao_cv2(erosao_cv2(img_bin, kernel), kernel)
        # fechamento = dilatacao + erosao
        img_close = erosao_cv2(dilatacao_cv2(img_open, kernel), kernel)

        return (img_close.astype(np.float32) / 255.0)[:, :, np.newaxis]


    # teste do Wrapper aplicando depois da funcoes


    result = tf.numpy_function(_apply, [image], tf.float32)
    result.set_shape(image.shape)
    return result


def pipeline_preProcessing(img, lbl):
    img = apply_salt_pepper(img)
    img = pipe_filtragem_espacial(img)
    img = pipe_morfologia(img)
    return img, lbl 