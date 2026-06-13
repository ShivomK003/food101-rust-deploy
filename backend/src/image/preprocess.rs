use anyhow::Result;
use image::{DynamicImage, ImageReader, imageops::FilterType};
use std::io::Cursor;

pub fn load_image(bytes: &[u8]) -> Result<DynamicImage> {
    let cursor = Cursor::new(bytes);

    let image = ImageReader::new(cursor).with_guessed_format()?.decode()?;

    Ok(image)
}

pub fn resize_and_crop(image: DynamicImage) -> DynamicImage {
    let (width, height) = (image.width(), image.height());

    let resized = if width < height {
        let new_height = ((height as f32) * 256.0 / (width as f32)) as u32;
        image.resize_exact(256, new_height, FilterType::Lanczos3)
    } else {
        let new_width = ((width as f32) * 256.0 / (height as f32)) as u32;
        image.resize_exact(new_width, 256, FilterType::Lanczos3)
    };

    let x = (resized.width() - 224) / 2;
    let y = (resized.height() - 224) / 2;

    resized.crop_imm(x, y, 224, 224)
}

pub fn image_to_tensor(image: &DynamicImage) -> Vec<f32> {
    let rgb_image = image.to_rgb8();

    let mean = [0.485_f32, 0.456_f32, 0.406_f32];
    let std = [0.229_f32, 0.224_f32, 0.225_f32];

    let mut tensor = vec![0.0_f32; 1 * 3 * 224 * 224];

    for y in 0..224 {
        for x in 0..224 {
            let pixel = rgb_image.get_pixel(x, y);

            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            let r = (r - mean[0]) / std[0];
            let g = (g - mean[1]) / std[1];
            let b = (b - mean[2]) / std[2];

            let index = (y * 224 + x) as usize;

            tensor[index] = r;
            tensor[224 * 224 + index] = g;
            tensor[2 * 224 * 224 + index] = b;
        }
    }

    tensor
}
