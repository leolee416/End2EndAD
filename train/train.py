import os
import torch
from drive import train_model
from torch.utils.data import DataLoader
from data_extractor import batch_generator
from model_1_output import ComplexCNN  # Assuming you have defined a complex CNN model in model.py
import torch.optim as optim
import torch.nn as nn
import time


def run_training(images_dir, drivings_dir, labels_dir, batch_size=32, epochs=10, learning_rate=0.001):
    """Run the training loop for the model"""
    
    # Initialize the model
    model = ComplexCNN().cuda()
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    

    m_image_files = [f for f in os.listdir(drivings_dir)]
    max_batches_per_epoch = len(m_image_files) // batch_size
    print(f"sizeofepoch:{max_batches_per_epoch}")

    
    # Start training
    print("Training started...")
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        # Initialize the data generator for batching
        train_data_gen = batch_generator(images_dir, drivings_dir, labels_dir, batch_size)
        running_loss = 0.0
        start_time = time.time()
        batch_count = 0


        # Iterate over the dataset in batches
        for i, (images, steering_angles, labels) in enumerate(train_data_gen):
            # if i >= max_batches_per_epoch:  # Stop if we've reached the max batch limit for the epoch
            #     break
            batch_count += 1
            images = torch.from_numpy(images).float().cuda()
            images = images.cuda()  # Move images to GPU
            images = images.permute(0, 3, 1, 2)
            steering_angles = torch.from_numpy(steering_angles).float().cuda()
            steering_angles = steering_angles.cuda()  # Move labels to GPU
            
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(images)  # Forward pass through the model
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), steering_angles)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights

            running_loss += loss.item()


            if (i+1) % 10 == 0:  # Print the loss every 10 batches
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}], Loss: {loss.item():.4f}")
        
        # Print the average loss for this epoch
        epoch_duration = time.time() - start_time
        print(f"batch_count: {batch_count}")
        if batch_count == 0:
            batch_count = 525
        print(f"Epoch [{epoch+1}/{epochs}] finished in {epoch_duration:.2f}s. Avg Loss: {running_loss / batch_count:.4f}")
    
    # Save the trained model after all epochs
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved as 'trained_model.pth'")

# Example usage
if __name__ == "__main__":
    images_dir = "Datasets_YangWu/images"
    drivings_dir = "Datasets_YangWu/drivings"
    labels_dir = "Datasets_YangWu/labels"
    
    run_training(images_dir, drivings_dir, labels_dir, batch_size=32, epochs = 40, learning_rate=0.0003)