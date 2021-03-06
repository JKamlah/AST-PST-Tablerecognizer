import sys
from string import digits
from pathlib import Path
import logging

from tesserocr import PyTessBaseAPI
import numpy as np
import cv2 as cv
from PIL import Image
import pandas as pd

logging.basicConfig(filename='error.log',format='%(asctime)s %(message)s', level=logging.WARNING)


def extract_tableinformation_from_images(img_paths, lang="ASTPST_0.522000_951_6100", whitelist_num=False,
                                         template=1, auto_template=True,
                                         suspicious_threshold=10,
                                         pixel_density_template_threshold=179,
                                         pixel_density_row_threshold=0.95,
                                         ocr_padding=15,
                                         debug=False, visual_debug=False):
    """
    This program process AST-PST files, it:
    - deskews via object clustering
    - analyzes the layout via 1d-projection profile and pixel density and templating
    - extract tableinformation via OCR with tesseract
    - stores the data as table in a csv-file
    For this special case magic number, thresholds and the templating are hardcoded.
    :return:
    """
    with PyTessBaseAPI(lang=lang, psm=4) as api:
        for img_path in img_paths:
            print(f"Processing {img_path}")
            try:
                # Read the image
                img = cv.imread(img_path, 0)

                # Binarize the image
                img = cv.medianBlur(img, 5)
                img = cv.medianBlur(img, 3)
                bin = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)

                # Dilate and erode to glue the borderelements and get a solid border
                kernel = np.ones((3, 3), np.uint8)
                bin = cv.dilate(bin, kernel, iterations=1)
                kernel = np.ones((2, 13), np.uint8)
                bin = cv.erode(bin, kernel, iterations=3)
                kernel = np.ones((13, 2), np.uint8)
                bin = cv.erode(bin, kernel, iterations=3)
                kernel = np.ones((5, 5), np.uint8)
                bin = cv.dilate(bin, kernel, iterations=8)

                # Get a dict with all {area:contours}
                areas = find_contour_areas(bin)

                # Find the biggest area which ends above the middle of the image
                for idx, (area, contour) in enumerate(reversed(sorted(areas.items()))):
                    x, y, w, h = cv.boundingRect(contour)
                    if x+h > 0.5*bin.shape[1]:
                        continue
                    # Draw the rectangle to see what you've found
                    #cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Region of interest
                    roi = bin[y:y+h, x:x+w]

                    # In each col search for the first black pixel and than calculate the deskewangle (tilt) for the image
                    y_idxs = []
                    for x in range(0 + 25, w - 25):
                        for y in range(0, h):
                            if roi[y, x] == 0:
                                y_idxs.append(y)
                                break

                    polyfit_value = np.polyfit(range(0, len(y_idxs)), y_idxs, 1)
                    deskewangle = np.arctan(polyfit_value[0]) * (360 / (2 * np.pi))

                    # Deskew the image and bin image
                    M = cv.getRotationMatrix2D((bin.shape[1] // 2, bin.shape[0] // 2), deskewangle, 1.0)
                    rotated_bin = cv.warpAffine(bin, M, (bin.shape[1], bin.shape[0]),
                                                flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
                    rotated_img = cv.warpAffine(img, M, (bin.shape[1], bin.shape[0]),
                                                flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
                    break
                else:
                    print("The roi in the original image could not be found.")
                    return

                # Again find the biggest area which ends above the middle of the image
                # (you could also calculate it from the results above.. :)
                areas = find_contour_areas(rotated_bin)
                for idx, (area, contour) in enumerate(reversed(sorted(areas.items()))):
                    x, y, w, h = cv.boundingRect(contour)
                    if x+h > 0.5*bin.shape[1]:
                        continue
                    # Region of interest
                    rotated_roi = bin[y:y + h, x:x + w]
                    if debug:
                        cv.rectangle(rotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    break
                else:
                    print("The roi in the rotated image could not be found.")
                    return

                # Set rotated img to tessapi
                rotated_img = cv.cvtColor(rotated_img, cv.COLOR_BGR2RGB)
                rotated_img_pil = Image.fromarray(rotated_img)
                api.SetImage(rotated_img_pil)

                # Get Land/-kreis approximately w*.2, h*0.6 above x,y
                api.SetRectangle(x, int(y-h*0.5), int(w*0.20), int(h*0.4))
                BL_LK = api.GetUTF8Text().replace('- ','-').strip().split('\n')
                if len(BL_LK) == 2:
                    bundesland = BL_LK[0]
                    landkreis = BL_LK[1]
                else:
                    bundesland = ""
                    landkreis = ' '.join(BL_LK)

                # Autodetect template with the pixel density of the roi?
                # Measurements of testfiles: 331 -> 182, 332 -> 140, 333 -> 180, 334-> 143
                if debug:
                    print('Pixel density: ',np.sum(rotated_roi)/area)
                if auto_template:
                    if np.sum(rotated_roi)/area > pixel_density_template_threshold:
                        template = 1
                    else:
                        template = 2

                # Cols: Now either use fix coordinates or find the columns with the rotated_roi
                cols = []
                if template == 1:
                    sep_vals = [580, 1055, 1210, 1395, 1620, 1830, 2035, 2240, 2445, 2610]
                elif template == 2:
                    sep_vals = [440, 580, 705, 875, 1045, 1245, 1445, 1645, 1875, 2080, 2290, 2490, 2690]
                for col in sep_vals:
                    col = int(w*col/sep_vals[-1])
                    cols.append(col)
                    if debug:
                        cv.line(rotated_img, (x+col, y+h), (x+col, img.shape[1]), (255, 0, 0), 3)

                # Rows: To get the rows you could use the 1D-projection of the rotated_bin
                projection = []
                th = 255*(rotated_bin.shape[1]-x-cols[1])*pixel_density_row_threshold
                kernel = np.ones((2, 13), np.uint8)
                rotated_bin_cols = cv.erode(rotated_bin, kernel, iterations=2)
                cv.imwrite("./rotated_cols.png", rotated_bin_cols)
                for yy in range(0, rotated_bin.shape[0]):
                    projection.append(th if sum(rotated_bin_cols[yy,cols[1]+x:]) > th else 0)
                projection_smooth = np.convolve(projection, 10, mode='same')
                minimas = np.diff(np.sign(np.diff(projection_smooth)))

                skip = False
                linepad = 15
                rows = []
                for idx, minima in enumerate(minimas):
                    if idx > y + h:
                        if minima == -1:
                            if not skip:
                                rows.append(idx-linepad)
                                if debug:
                                    cv.line(rotated_img, (0, idx-linepad), (img.shape[1], idx-linepad), (255, 0, 0), 3)
                                skip = True
                            else:
                                skip = False
                cv.imwrite("./debug.png", rotated_img)

                # Create pandas (pd) dataframe (df) for the templates
                if template==1:
                    tmpl_header = ["Bundesland",
                                   "Landkreis",
                                   "Wirtschaftszweig",
                                   "Sypro",
                                   "Pst_n",
                                   "Beschaeftigte",
                                   "Grundmittel",
                                   "Gebaeude",
                                   "Ausruestungen",
                                   "Grundstueck_Flaeche",
                                   "Gebaeude_Flaeche",
                                   "Gleisanschluss",
                                   "Dateiname",
                                   "Fulltext"]
                else:
                    tmpl_header = ["Bundesland",
                                   "Landkreis",
                                   "Wirtschaftszweig",
                                   "Sypro",
                                   "Pst_n",
                                   "Personal_insg",
                                   "Personal_mehrsch",
                                   "Waren_prod",
                                   "Produktion_flaeche",
                                   "Elektro_verbrauch",
                                   "Waerme_verbrauch",
                                   "Waerme_erzeugung",
                                   "Wasser_verbrauch",
                                   "Wasser_erzeugung",
                                   "Wasser_verwendung",
                                   "Dateiname",
                                   "Fulltext"]


                df = pd.DataFrame(columns=tmpl_header)

                # OCR: Use tesseract via tesserocr and the roi

                suspicious_counter = 0
                rows.append(rotated_img.shape[0]-10-ocr_padding)
                whitelist = api.GetVariableAsString('tessedit_char_whitelist')
                for ridx, row in enumerate(rows[1:]):
                    print(f"Zeile: {ridx+1}")
                    if whitelist_num:
                        api.SetVariable('tessedit_char_whitelist', whitelist)
                    df_row = []
                    df_row.append(bundesland)
                    df_row.append(landkreis)
                    x_start = x
                    y_start = rows[ridx+1-1]
                    y_end = row
                    # Read the full row

                    api.SetRectangle(x_start - ocr_padding, y_start - ocr_padding, x_start+w+ocr_padding,
                                     y_end - y_start + ocr_padding)
                    fullrow_text = api.GetUTF8Text().strip().replace('- ','-')
                    print('Fulltext: ',fullrow_text)
                    for cidx, col in enumerate(cols):
                        print(f"Spalte: {cidx+1}")
                        if whitelist_num:
                            # Use for col > 0 only digits as allowed chars
                            if cidx > 0:
                                api.SetVariable('tessedit_char_whitelist', digits)

                        if cidx > 0:
                            col = col-cols[cidx-1]
                        api.SetRectangle(x_start-ocr_padding, y_start-ocr_padding,
                                         col+ocr_padding, y_end-y_start+ocr_padding)
                        if debug and False:
                            crop = rotated_img_pil.crop([x_start-ocr_padding, y_start-ocr_padding,
                                                         col+ocr_padding+(x_start-ocr_padding),
                                                         y_end-y_start+ocr_padding+(y_start-ocr_padding)])
                            crop.save(f"./{ridx}_{cidx}.png")
                        ocrResult = api.GetUTF8Text().replace('\n', ' ').replace('- ','-').strip()
                        if ocrResult == "":
                            # Set psm mode to single character
                            api.SetPageSegMode(10)
                            ocrResult = api.GetUTF8Text().replace('\n', ' ').strip()
                            # Check if the single char is a digit else use empty text
                            ocrResult = ocrResult if ocrResult.isdigit() else ''
                            # Reset to psm mode singe column
                            api.SetPageSegMode(4)
                        # Find suspicious ocr results when checking non-empty ocr-results with the result of the
                        # row fulltext (0 not included cause it often gets not recognized by in the row fulltext)
                        if len(cols)-1 > cidx > 0 and ocrResult not in ['', '0'] and (not ocrResult.isdigit() or
                                                                           ocrResult not in fullrow_text):
                            ocrResult += " ?!?"
                            suspicious_counter += 1
                        elif len(cols)-1==cidx and ocrResult != "" and not ocrResult.isdigit():
                            ocrResult += " ?!?"
                            suspicious_counter += 1
                        print(ocrResult)
                        x_start += col
                        df_row.append(ocrResult)
                    df_row.append(Path(img_path).name)
                    df_row.append(fullrow_text)
                    df.loc[len(df)+1] = df_row

                # Et voil??: Deskew images, ocr results and extracted the information
                df.to_csv(Path(img_path).with_suffix('.csv'), index=False)
                df = None
                if suspicious_counter > suspicious_threshold:
                    logging.warning(f'More than 10 suspicious occurrences in {img_path}')
                # Plot the results (for debugging)
                if visual_debug:
                    from matplotlib import pyplot as plt
                    plt.subplot(1,2,1), plt.imshow(roi, cmap='gray')
                    plt.subplot(1,2,2), plt.imshow(rotated_img, cmap='gray')
                    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                    plt.show()

            except Exception as excp:
                # Log all files who could not be processed properly
                logging.error(f'Error while processing: {img_path}\n\t\tError:{excp}\n')


def find_contour_areas(bin):
    # Find contours
    contours, hierachies = cv.findContours(bin.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Find the convex hull object for each contour
    areas = {}
    size_th = 0.5*bin.size
    for idx, contour in enumerate(contours):
        cnt_area = cv.contourArea(contour, False)
        if cnt_area < size_th:
            areas[cnt_area] = contour
    return areas

if __name__ == '__main__':
    img_list = []
    for idx, img_path in enumerate(reversed(sys.argv[1:])):
        if Path(img_path).is_dir():
            img_list.extend([str(img_path.resolve()) for img_path in Path(img_path).rglob('*.png')])
        else:
            img_list.append(str(Path(img_path).resolve()))
    extract_tableinformation_from_images(list(set(img_list)))
