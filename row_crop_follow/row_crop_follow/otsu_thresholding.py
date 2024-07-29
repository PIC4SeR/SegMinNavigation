import numpy as np
import cv2


def show_image(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(1)


def bgr2gray_2gbr(img):
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    gray = 2 * g - b - r
    return np.clip(gray, 0, 255).astype(np.uint8)


def bgr2gray_gb(img):
    b = img[:, :, 0]
    g = img[:, :, 1]
    return np.where(g > b, g - b, 0).astype(np.uint8)


class OtsuThresholding:

    def __init__(
        self,
        proportional_criterion=1.5,
        deviation_control_gain=0.01,
        correction_control_gain=-0.05,
        linear_velocity=0.5,
        grayscale_method="cv2",
    ):
        self.PC = proportional_criterion
        self.G = deviation_control_gain
        self.P = correction_control_gain
        self.V = linear_velocity
        self.grayscale_method = grayscale_method

    def run(self, img, show_result=False):
        if not isinstance(img, np.ndarray):
            raise ValueError("img must be a numpy array")

        img_binary = self.__binarize(img)
        img_binary = self.__keep_only_biggest_area(img_binary)
        central_lane_pixel = np.argmax(np.sum(img_binary, axis=0))

        white_L = np.sum(img_binary[:, :central_lane_pixel])
        white_R = np.sum(img_binary[:, central_lane_pixel:])

        error = central_lane_pixel - img_binary.shape[1] / 2

        if not (white_L > 0 and white_R > 0):
            return 0.0, 0.0, "NO LANE", img

        if white_L / white_R > self.PC or white_R / white_L > self.PC:
            # CORRECTION CONTROL
            control = "CORRECTION"
            w = self.P * error
        else:
            # DEVIATION CONTROL
            control = "DEVIATION"
            w = self.G * error

        result = img.copy()
        result = cv2.line(result, (central_lane_pixel, 0), (central_lane_pixel, result.shape[0]), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Error: {error:.1f}, v: {self.V:.2f}, w: {w:.2f}"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        textX = int((img.shape[1] - text_size[0]) / 2)
        textY = int((img.shape[0] + text_size[1]) / 2)

        result = cv2.putText(
            result,
            text,
            (textX, img.shape[0] - text_size[1]),
            font,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        if show_result:
            show_image("result", result)

        return self.V, w, control, result

    def __keep_only_biggest_area(self, img_binary, show_result=False):
        src = img_binary.copy()
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours[1:]:
            img_binary = cv2.drawContours(img_binary, [contour], -1, 0, -1)
        if show_result:
            show_image(cv2.drawContours(cv2.cvtColor(src, cv2.COLOR_GRAY2BGR), contours, -1, (0, 0, 255), 3))
            show_image(img_binary)
        return img_binary

    def __binarize(self, img, show_result=False):
        match self.grayscale_method:
            case "cv2":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            case "2bgr":
                gray = bgr2gray_2gbr(img)
            case "gb":
                gray = bgr2gray_gb(img)

        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        otsu_threshold, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if show_result:
            result = np.concatenate([gray, blur, otsu], axis=1)
            show_image(result, "binary image")
        return otsu

    def on_shutdown(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass
