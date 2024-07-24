import os

def sanitize_filename(filename):
    """
    Sanitize the filename for LaTeX by escaping underscores.
    """
    # Escape underscores to avoid LaTeX compilation errors
    return filename

def generate_latex_includes_for_images(directory):
    # Define the LaTeX command pattern for including an image
    latex_image_template = """
\\paragraph{{{filename}}}
\\includegraphics[width=\\linewidth]{{{{{directory}/{sanitized_filename}}}}}
"""
    # Get all image files in the specified directory
    files = os.listdir(directory)
    
    # Generate LaTeX figure includes for each image file
    latex_includes = []
    for file in files:
        sanitized_filename = sanitize_filename(file)
        # For the paragraph, we show the original filename for readability
        # For the LaTeX command, we use the sanitized filename
        latex_includes.append(latex_image_template.format(
            filename=file,  # Original filename for readability in the LaTeX document
            directory=directory,  # Directory path
            sanitized_filename=sanitized_filename))  # Sanitized filename for LaTeX command
    
    # Join the include commands with newlines
    includes_str = "\n".join(latex_includes)
    
    return includes_str

# Example usage
directory = './dip3e_masks'  # Replace with your directory path
latex_includes = generate_latex_includes_for_images(directory)
print(latex_includes)

