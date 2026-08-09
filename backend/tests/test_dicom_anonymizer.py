from pathlib import Path
from uuid import uuid4
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID

from app.infra.imaging.dicom_anonymizer import anonymize_dicom_series
from app.infra.imaging.dicom_inventory import DicomSeriesRecord

def test_dicom_anonymizer(tmp_path):
    # Setup dummy DICOM file
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
    file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
    file_meta.ImplementationClassUID = UID("1.2.3.4")
    
    ds = FileDataset(str(tmp_path / "dummy.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "John Doe"
    ds.PatientBirthDate = "19900101"
    ds.PatientID = "123456789"
    ds.InstitutionName = "Test Hospital"
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    
    dummy_path = tmp_path / "dummy.dcm"
    ds.save_as(str(dummy_path))
    
    # Setup record
    record = DicomSeriesRecord(
        study_instance_uid="1",
        series_instance_uid="2",
        modality="MR",
        series_description="test",
        protocol_name="test",
        image_type=("ORIGINAL", "PRIMARY"),
        manufacturer="Test",
        manufacturer_model_name="Test",
        magnetic_field_strength=3.0,
        pixel_spacing=(1.0, 1.0),
        slice_thickness=1.0,
        spacing_between_slices=1.0,
        orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        rows=256,
        columns=256,
        files=(dummy_path,),
        extra_metadata={},
    )
    
    # Run anonymizer
    output_dir = tmp_path / "anonymized"
    patient_uuid = uuid4()
    
    anonymized_files = anonymize_dicom_series(record, output_dir, patient_uuid)
    
    assert len(anonymized_files) == 1
    assert anonymized_files[0].exists()
    
    # Verify anonymization
    anon_ds = pydicom.dcmread(str(anonymized_files[0]))
    assert anon_ds.PatientName == ""
    assert anon_ds.PatientBirthDate == ""
    assert anon_ds.InstitutionName == ""
    assert anon_ds.PatientID == str(patient_uuid)
