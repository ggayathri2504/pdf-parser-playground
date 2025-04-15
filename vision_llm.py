import os
import base64
import asyncio
import shutil
import logging
from typing import List
import fitz
from openai import AsyncOpenAI
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Then change your convert_from_path code to:
def pdf_to_images(pdf_path, output_dir):
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(alpha=False)
        image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    
    return image_paths

class DocumentProcessor:
    def __init__(self, openai_key: str,max_concurrent_requests: int = 5):
        self.client = AsyncOpenAI(api_key=openai_key)
        self.MODEL = "gpt-4o"
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        logger.info(
            "DocumentProcessor initialized with max_concurrent_requests=%d",
            max_concurrent_requests,
        )

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                logger.info("Encoding image: %s", image_path)
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error("Error encoding image %s: %s", image_path, str(e))
            raise

    async def process_single_image(self, image_path: str) -> str:
        """Process a single image with rate limiting."""
        async with self.semaphore:
            try:
                base64_image = self.encode_image(image_path)
                logger.info("Processing image: %s", image_path)

                response = await self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that responds in Markdown.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Convert the following PDF page to markdown. "
                                        "Return only the markdown with no explanation text. "
                                        "Do not exclude any content from the page. "
                                        "Do not include delimiters like ```markdown or ```. "
                                        "\n\nReplace images with brief [descriptive summaries], "
                                        "and use appropriate markdown syntax (headers [#, ##, ###, ####], "
                                        "bold **, italic *). Output should be clean, formatted markdown "
                                        "that matches the original layout."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ],
                        },
                    ],
                    temperature=0.0,
                )
                logger.info("Processing completed for image: %s", image_path)
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error("Error processing image %s: %s", image_path, str(e))
                return f"Error processing page: {str(e)}"

    async def process_document(self, file_path: str, filename: str) -> str:
        """Process an entire document concurrently."""
        image_dir = f"file_{os.path.splitext(filename)[0]}_images"
        os.makedirs(image_dir, exist_ok=True)

        try:
            logger.info("Processing document: %s", file_path)

            if file_path.lower().endswith(".pdf"):
                logger.info("Converting PDF to images: %s", file_path)
                # images = convert_from_path(file_path)
                image_paths: List[str] = []
                image_paths = pdf_to_images(file_path, image_dir)

                # for i, image in enumerate(images):
                #     image_path = os.path.join(image_dir, f"page_{i + 1}.png")
                #     image.save(image_path, "PNG")
                #     image_paths.append(image_path)
                logger.info("Saved page as image", image_dir)
            else:
                # For single image files
                image_paths = [file_path]

            # Process images concurrently
            tasks = [self.process_single_image(path) for path in image_paths]
            results = await asyncio.gather(*tasks)

            logger.info("Document processing complete: %s", file_path)
            return "\n\n---\n\n".join(results)  # Add separator between pages if needed

        finally:
            logger.info("Cleaning up directory: %s", image_dir)
            shutil.rmtree(image_dir)
