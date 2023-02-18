import random

import cv2
import numpy as np

from main import get_image


styles = """
<style>
    .small {
      font: 7px monospace;
    }
  </style>
"""

animation = """
<animate
    attributeName="rotate"
    from="0" to="360"
    begin="0s" dur="10s"
    repeatCount="indefinite">
</animate>
"""


def make():
    target = get_image("images/target.jpeg", (700, 700))
    # target = cv2.GaussianBlur(target, (7, 7), 0)

    im_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)

    cv2.imshow("path.png", thresh)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

    blank = np.zeros(target.shape, dtype=np.uint8)
    cv2.drawContours(blank, contours, -1, (0, 255, 0), 3)
    cv2.imshow("path.png", blank)
    cv2.waitKey(0)

    h, w, _ = target.shape

    text = "happy anniversary to you"

    with open("path.svg", "w+") as f:
        f.write(f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"><defs>')
        f.write(styles)

        lengths = []

        for i, contour in enumerate(contours):
            lengths.append(cv2.arcLength(contour, True))

            if lengths[-1] < 12 or lengths[-1] > 2000:
                continue

            f.write(f'<path id="path{i}" d="M')
            for point in contour:
                x, y = point[0]
                f.write(f"{x} {y} ")
            f.write('"/>')

        f.write("</defs>")

        for i, length in enumerate(lengths):
            length = round(length)

            if length < 12 or length > 2000:
                continue

            if len(text) < length:
                text += text * ((length - len(text)) // len(text))

            f.write(f'<text class="small"><textPath href="#path{i}" >')
            f.write(text)
            f.write(f"</textPath> {animation if length < 5000 else ''}</text>")

        f.write("</svg>")


if __name__ == "__main__":
    make()
