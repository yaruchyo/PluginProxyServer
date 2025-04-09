import os
from proxy_package.utils.logger import logger
from proxy_package.domain_layer.file_responce import Response

def save_files_from_response(structured_response: Response, base_dir: str = "."):
    """
    Processes the files_to_update list from a Response object
    and writes the code content to the specified files relative to base_dir.

    Args:
        structured_response: The Response object containing file update information.
        base_dir: The base directory where files should be saved. Defaults to current dir.
    """
    if not structured_response.files:
        logger.info("No files to update found in the response.")
        return

    logger.info(f"Processing {len(structured_response.files)} file(s) for update...")

    for file_info in structured_response.files:
        # --- Input Validation ---
        if not file_info.filename:
            logger.warning("Skipping file update: 'filename' is missing or empty.")
            continue
        # We check for None specifically, allowing empty string "" as valid content
        if file_info.code is None:
            logger.warning(f"Skipping file update for '{file_info.filename}': 'code_to_update' is missing (None).")
            continue

        # --- Path Construction and Safety ---
        # Clean the filename path (e.g., remove './', handle '../')
        relative_path = os.path.normpath(file_info.filename)

        # Basic security check: prevent writing outside the intended base_dir
        # Disallow absolute paths and paths trying to escape the base directory
        if os.path.isabs(relative_path) or relative_path.startswith(".."):
            logger.warning(f"Skipping potentially unsafe file path: '{file_info.filename}'")
            continue

        # Construct the full path relative to the base directory
        full_path = os.path.join(base_dir, relative_path)

        logger.info(f"Attempting to write file: {full_path}")

        try:
            # --- Directory Creation ---
            # Ensure the target directory exists before writing the file
            directory = os.path.dirname(full_path)
            if directory:  # Only create if directory part is not empty
                # exist_ok=True prevents an error if the directory already exists
                os.makedirs(directory, exist_ok=True)
                # logger.debug(f"Ensured directory exists: {directory}") # Optional: uncomment for verbose logging

            # --- File Writing ---
            # Open the file in write mode ('w'). This will overwrite existing files.
            # Use 'utf-8' encoding as a standard practice.
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(file_info.code)
            logger.info(f"Successfully wrote {len(file_info.code)} bytes to {full_path}")

        except OSError as e:
            # Handle potential OS errors like permission issues, invalid paths etc.
            logger.error(f"Error writing file {full_path}: {e}")
        except Exception as e:
            # Catch any other unexpected errors during file processing
            logger.error(f"An unexpected error occurred while processing {full_path}: {e}")
            logger.exception(e) # Log traceback for unexpected errors

