import numpy as np
import cv2


## funcoes da morfologia da m2


   #Erosão
def erosao(img, kernel, objeto="preto"):
    
    """
    Aplica erosão em uma imagem limiarizada.

    objeto:
    - "branco": considera pixels brancos
    - "preto": considera pixels pretos
    """

    img = (img > 0).astype(np.uint8)
    k = (kernel > 0).astype(np.uint8)

    altura, largura = img.shape
    kh, kw = k.shape

    pad_y = kh // 2
    pad_x = kw // 2

    if objeto == "branco":
        valor_objeto = 1
        valor_fundo = 0

    elif objeto == "preto":
        valor_objeto = 0
        valor_fundo = 1

    #padding com valor de fundo
    img_padded = np.pad(
        img,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="constant",
        constant_values=valor_fundo
    )

    #Resultado começa preenchido com fundo
    resultado = np.full_like(img, valor_fundo, dtype=np.uint8)

    #percorre a imagem
    for y in range(altura):
        for x in range(largura):

            img_igual_kernel = True

            #percorre o kernel
            for i_kernel in range(kh):
                for j_kernel in range(kw):

                    if k[i_kernel, j_kernel] == 1:

                        pixel = img_padded[y + i_kernel, x + j_kernel]

                        #confere se o pixel não bate com o kernel
                        if pixel != valor_objeto:
                            img_igual_kernel = False

            if img_igual_kernel == True:
                resultado[y, x] = valor_objeto
            else:
                resultado[y, x] = valor_fundo

    resultado = resultado * 255
    resultado = resultado.astype(np.uint8)

    return resultado


#Dilatação
def dilatacao(img, kernel, objeto="preto"):
    """
    Aplica dilatação em uma imagem limiarizada.

    objeto:
    - "branco": considera pixels brancos
    - "preto": considera pixels pretos
    """

    img = (img > 0).astype(np.uint8)
    k = (kernel > 0).astype(np.uint8)

    altura, largura = img.shape
    kh, kw = k.shape

    pad_y = kh // 2
    pad_x = kw // 2

    if objeto == "branco":
        valor_objeto = 1
        valor_fundo = 0

    elif objeto == "preto":
        valor_objeto = 0
        valor_fundo = 1

    #padding com valor de fundo
    img_padded = np.pad(
        img,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="constant",
        constant_values=valor_fundo
    )

    #Resultado começa preenchido com fundo
    resultado = np.full_like(img, valor_fundo, dtype=np.uint8)

    #percorre a imagem
    for y in range(altura):
        for x in range(largura):

            encontrou_objeto = False

            #percorre o kernel
            for i_kernel in range(kh):
                for j_kernel in range(kw):

                    if k[i_kernel, j_kernel] == 1:

                        pixel = img_padded[y + i_kernel, x + j_kernel]

                        #confere se o pixel não bate com o kernel
                        if pixel == valor_objeto:
                            encontrou_objeto = True

            if encontrou_objeto == True:
                resultado[y, x] = valor_objeto
            else:
                resultado[y, x] = valor_fundo

    resultado = resultado * 255
    resultado = resultado.astype(np.uint8)

    return resultado

# ---- versoes vetorizadas com cv2 (mesma logica, mas em C) ----
# O cv2 trata BRANCO como objeto. Como nossa mao e PRETA (objeto="preto"),
# invertemos a imagem antes (mao vira branca), aplicamos, e invertemos de volta.

def erosao_cv2(img, kernel, objeto="preto"):
    if objeto == "preto":
        img = cv2.bitwise_not(img)
    resultado = cv2.erode(img, kernel)
    if objeto == "preto":
        resultado = cv2.bitwise_not(resultado)
    return resultado


def dilatacao_cv2(img, kernel, objeto="preto"):
    if objeto == "preto":
        img = cv2.bitwise_not(img)
    resultado = cv2.dilate(img, kernel)
    if objeto == "preto":
        resultado = cv2.bitwise_not(resultado)
    return resultado


def apply_limiar(img_in, limiar):
    
    img_in = img_in.copy()
    img_out = np.zeros(img_in.shape)

    for i in range(img_in.shape[0]):
        for j in range(img_in.shape[1]):
            if img_in[i, j] > limiar:
                img_out[i, j] = 255
            else:
                img_out[i, j] = 0

    
    img_out = img_out.astype(np.uint8)

    return img_out



def abertura(img, kernel, objeto="branco"):
    """
    Aplica abertura em uma imagem limiarizada.

    Abertura = erosão + dilatação.
    """
    img_erodida = erosao(img, kernel, objeto)
    img_aberta = dilatacao(img_erodida, kernel, objeto)

    return img_aberta


def fechamento(img, kernel, objeto="branco"):
    """
    Aplica fechamento em uma imagem limiarizada.

    Fechamento = dilatação + erosão.
    """
    img_dilatada = dilatacao(img, kernel, objeto)
    img_fechada = erosao(img_dilatada, kernel, objeto)

    return img_fechada
