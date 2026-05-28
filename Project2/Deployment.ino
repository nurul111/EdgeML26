// Correct for modern TFLite Micro
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>

// Camera Library
#include <Arduino_OV767X.h>

// Your local working directory files
#include "model.h"

// Allocate 15 KB memory buffer for model processing execution
const int tensor_arena_size = 15 * 1024;
alignas(16) uint8_t tensor_arena[tensor_arena_size]; 

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

const char* labels[] = {"rock", "paper", "scissors"};

// Camera configuration
const int cam_width = 160;  // QQVGA width
const int cam_height = 120; // QQVGA height
uint16_t cam_buffer[160 * 120]; // Buffer to hold raw RGB565 camera frame (38.4 KB)

void setup() {
  Serial.begin(115200);
  while (!Serial); // Wait for Serial Monitor 
  
  Serial.println(F("========================================="));
  Serial.println(F("TinyML Session 5: Live Camera Deployment "));
  Serial.println(F("========================================="));

  // --- 1. Initialize Camera ---
  if (!Camera.begin(QQVGA, RGB565, 1)) {
    Serial.println(F("❌ Critical Error: Failed to initialize OV767X camera!"));
    while (1);
  }
  Serial.println(F("📷 OV767X Camera Initialized."));

  // --- 2. Initialize TFLite ---
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
    tflite::GetModel(model), micro_op_resolver, tensor_arena, tensor_arena_size
  );
  interpreter = &static_interpreter;

  // Zero-argument allocation for the official TFLite library
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("❌ Critical Error: Tensor allocation failed!"));
    while (1);
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  Serial.println(F("🧠 TFLite Model loaded successfully. Ready for live inferences."));
}

void loop() {
  // 1. Capture Frame 
  Camera.readFrame(cam_buffer);
    
  // 2. Preprocess: Crop to center 120x120, Downsample to 32x32, Convert to Grayscale
  int8_t* input_buffer = input_tensor->data.int8;
  int target_width = 32;
  int target_height = 32;
  int crop_size = 120; 
  int offset_x = (cam_width - crop_size) / 2; 
  
  for (int y = 0; y < target_height; y++) {
    for (int x = 0; x < target_width; x++) {
      int src_x = offset_x + (x * crop_size / target_width);
      int src_y = (y * crop_size / target_height);
      
      uint16_t pixel = cam_buffer[src_y * cam_width + src_x];
      
      uint8_t r = (pixel & 0xF800) >> 8;
      uint8_t g = (pixel & 0x07E0) >> 3;
      uint8_t b = (pixel & 0x001F) << 3;
      
      uint8_t gray = (r * 77 + g * 150 + b * 29) >> 8;
      input_buffer[y * target_width + x] = (int8_t)(gray - 128);
    }
  }

  // 3. Run Inference
  if (interpreter->Invoke() == kTfLiteOk) {
    // 4. Output Results as Percentages!
    Serial.println(F("\n--- LIVE INFERENCE ---"));
    
    for (int i = 0; i < 3; i++) {
      // Grab the raw signed int8 value from memory
      int8_t raw_score = output_tensor->data.int8[i];
      
      // Mathematically shift it from (-128 to 127) into a (0 to 100) percentage
      int percentage = ((raw_score + 128) * 100) / 255;
      
      // Print the clean results
      Serial.print(labels[i]);
      Serial.print(F(": "));
      Serial.print(percentage);
      Serial.println(F("%")); 
    }
    Serial.println(F("----------------------"));
  } else {
    Serial.println(F("⚠️ Warning: Model invocation failed this frame."));
  }
  
  delay(1000); 
}