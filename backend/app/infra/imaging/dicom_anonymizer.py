import os
from pathlib import Path
from uuid import UUID
import pydicom
from app.core.audit import log_audit_event

def anonymize_dicom_series(record, output_dir: Path, patient_uuid: UUID) -> tuple[Path, ...]:
    """
    Anonymize a DICOM series by stripping targeted PHI tags and replacing PatientID.
    
    Args:
        record: The DicomSeriesRecord containing the file paths.
        output_dir: The directory where anonymized files should be written.
        patient_uuid: The UUID to use as the pseudonymized PatientID.
        
    Returns:
        A tuple of paths to the anonymized DICOM files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    anonymized_files = []
    
    # Target tags to scrub (PN: Patient Name, DA: Patient Birth Date, etc.)
    # We use targeted string representation of tags to avoid blindly scrubbing all LO/DA tags.
    TARGET_TAGS_TO_BLANK = [
        "PatientName",
        "PatientBirthDate",
        "OtherPatientNames",
        "PatientBirthTime",
        "PatientSex",
        "PatientAge",
        "PatientWeight",
        "PatientAddress",
        "InstitutionName",
        "InstitutionAddress",
        "InstitutionalDepartmentName",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorsName",
    ]
    
    for file_path in record.files:
        try:
            dataset = pydicom.dcmread(str(file_path), force=True)
            
            # Blank out specific PHI tags
            for tag_name in TARGET_TAGS_TO_BLANK:
                if tag_name in dataset:
                    elem = dataset.data_element(tag_name)
                    if elem.VR == "PN":
                        dataset[tag_name].value = ""
                    else:
                        # For DA, AS, SH, LO etc.
                        dataset[tag_name].value = ""
            
            # Pseudonymize Patient ID
            if "PatientID" in dataset:
                dataset.PatientID = str(patient_uuid)
            else:
                # Add PatientID if it doesn't exist
                dataset.add_new((0x0010, 0x0020), 'LO', str(patient_uuid))
                
            # Write anonymized file
            output_file_path = output_dir / file_path.name
            dataset.save_as(str(output_file_path))
            anonymized_files.append(output_file_path)
            
        except Exception as e:
            # If a file cannot be read or anonymized, we can log it and skip or raise
            raise RuntimeError(f"Failed to anonymize {file_path}: {e}")
            
    log_audit_event(
        action="ANONYMIZE_DICOM",
        resource_id=str(patient_uuid),
        details={"file_count": len(anonymized_files), "series_uid": record.series_instance_uid}
    )
            
    return tuple(anonymized_files)
