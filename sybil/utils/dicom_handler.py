import logging
import os
from typing import List

import numpy as np
import pydicom

from config import PYTHON_ENV

from .config import VISUALIZATION_CONFIG as cfg


class DicomHandler:
    @staticmethod
    def save_overlay_as_dicom(
        image: np.ndarray,
        original_metadata: pydicom.Dataset,
        save_path: str,
        base_filename: str,
    ) -> bool:
        """
        Save an overlay image as a DICOM file.

        Args:
            image: The overlay image to save (RGB format)
            original_metadata: Original DICOM metadata to copy from
            save_path: Directory to save the DICOM file
            base_filename: Base name for the output file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a copy of the original metadata
            ds = original_metadata.copy()

            # Convert image to uint8 (0-255)
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)

            # Configure metadata for RGB DICOM
            ds.Rows, ds.Columns = img_uint8.shape[:2]
            ds.SamplesPerPixel = cfg["DICOM"]["SAMPLES_PER_PIXEL"]
            ds.PhotometricInterpretation = cfg["DICOM"]["PHOTOMETRIC_INTERPRETATION"]
            ds.BitsAllocated = cfg["DICOM"]["BITS_ALLOCATED"]
            ds.BitsStored = cfg["DICOM"]["BITS_STORED"]
            ds.HighBit = cfg["DICOM"]["HIGH_BIT"]
            ds.PlanarConfiguration = cfg["DICOM"]["PLANAR_CONFIGURATION"]
            ds.PixelRepresentation = cfg["DICOM"]["PIXEL_REPRESENTATION"]
            ds.RescaleIntercept = cfg["DICOM"]["RESCALE_INTERCEPT"]
            ds.RescaleSlope = cfg["DICOM"]["RESCALE_SLOPE"]
            ds.VOILUTFunction = cfg["DICOM"]["VOI_LUT_FUNCTION"]

            # Copy important metadata attributes
            for attr in cfg["DICOM"]["ATTRIBUTES_TO_COPY"]:
                if hasattr(original_metadata, attr):
                    setattr(ds, attr, getattr(original_metadata, attr))

            # Set pixel data
            ds.PixelData = img_uint8.tobytes()

            # Save DICOM file
            dicom_path = os.path.join(save_path, f"{base_filename}.dcm")
            ds.save_as(dicom_path)
            logging.info(f"✅ Successfully saved DICOM overlay: {dicom_path}")
            if PYTHON_ENV == "develop":
                print(f"✅ Saved DICOM overlay (RGB): {dicom_path}")
            return True

        except Exception as e:
            logging.error(f"⚠️ Error saving DICOM file: {str(e)}")
            if PYTHON_ENV == "develop":
                print(f"⚠️ Error saving DICOM file: {str(e)}")
            return False
