import numpy as np
import tensorflow as tf
import cv2


# ---- copiado de m1_processing.py ----

def convolution(img, kernel):
    return cv2.filter2D(img, -1, kernel)




def gauss_create(sigma, size):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    exp = np.exp(-(((x**2) + (y**2)) / (2 * (sigma**2))))
    return (exp / exp.sum()).astype(np.float32)


def unsharp_mask(img, blur_img, fr):
    img = np.array(img, dtype=np.float32)
    blur_img = np.array(blur_img, dtype=np.float32)
    mask = img - blur_img
    sharp = img + (fr * mask)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def gamma_correction(img, gamma):
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    return gamma_corrected


# ---- wrappers para TF ----
# tf.numpy_function permite executar código numpy dentro do pipeline TF. executa no .map()

def median_blur(image, size=3):
    def _apply(img_np):
        img_np = (img_np * 255).astype(np.uint8)
        channels = [cv2.medianBlur(img_np[:, :, c], size) for c in range(img_np.shape[2])]
        result = np.stack(channels, axis=2).astype(np.float32) / 255.0
        return result

    result = tf.numpy_function(_apply, [image], tf.float32)
    result.set_shape(image.shape)
    return result



def apply_gamma(image, gamma=1.5):
    def _apply(img_np):
        img_np = (img_np * 255).astype(np.uint8)
        result = gamma_correction(img_np, gamma).astype(np.float32) / 255.0
        return result


    ## Wrap
    result = tf.numpy_function(_apply, [image], tf.float32)
    result.set_shape(image.shape)
    return result


def gaussian_blur(image, sigma=1.0, size=5):
    kernel = gauss_create(sigma, size).astype(np.float32)

    def _apply(img_np):
        img_np = (img_np * 255).astype(np.uint8)
        channels = [convolution(img_np[:, :, c], kernel) for c in range(img_np.shape[2])]
        result = np.stack(channels, axis=2).astype(np.float32) / 255.0
        return result


    #tecnica para aplicar esse processamento no .map(), o tf só aplica quando o dataset for p o processamento
    
    
    result = tf.numpy_function(_apply, [image], tf.float32)
    result.set_shape(image.shape)
    return result


def sharpening(image, amount=1.0, sigma=1.0, size=3):
    blurred = gaussian_blur(image, sigma, size)

    def _apply(img_np, blr_np):
        img_np = (img_np * 255).astype(np.uint8)
        blr_np = (blr_np * 255).astype(np.uint8)
        channels = [unsharp_mask(img_np[:, :, c], blr_np[:, :, c], amount) for c in range(img_np.shape[2])]
        result = np.stack(channels, axis=2).astype(np.float32) / 255.0
        return result

    result = tf.numpy_function(_apply, [image, blurred], tf.float32)
    result.set_shape(image.shape)
    return result
