import sys
import grpc
import torch
import parameter_server_pb2_grpc, parameter_server_pb2
from model import initialize_model
from serializers import serialize_gradients, deserialize_gradients, deserialize_model_weights
import pickle

def receive_model_weights(stub, worker_id):
    """Receive and reconstruct model weights from the parameter server."""
    request_iterator = iter([parameter_server_pb2.ModelWeightsRequest(worker_id=worker_id)])
    model_weights_chunks = stub.DistributeModelWeights(request_iterator)
    model_weights = b''
    for chunk in model_weights_chunks:
        model_weights += chunk.model_weights_chunk
        if chunk.is_final_chunk:
            break
    return deserialize_model_weights(model_weights)

def fetch_dataset_portion(stub, worker_id):
    """Fetch a portion of the dataset from the parameter server."""
    request = parameter_server_pb2.DatasetRequest(worker_id=worker_id)
    response = stub.DistributeDataset(request)
    if response:
        dataset_portion = pickle.loads(response.dataset_portion)
        is_end_of_dataset = response.is_end_of_dataset
        return dataset_portion, is_end_of_dataset
    else:
        raise Exception("Failed to fetch dataset portion")

def generate_gradient_chunks(gradients, step_count, chunk_size=1024 * 1024):
    """Yield serialized gradient chunks."""
    serialized_gradients = serialize_gradients(gradients)
    for i in range(0, len(serialized_gradients), chunk_size):
        yield parameter_server_pb2.GradientsRequest(
            worker_id=worker_id,
            gradients=serialized_gradients[i:i+chunk_size],
            training_step=step_count
        )

def receive_and_apply_averaged_gradients(stub, worker_id, step_count, model):
    """Receive streamed averaged gradients and apply them to the model."""
    request = parameter_server_pb2.AveragedGradientsRequest(training_step=step_count)
    averaged_gradients_stream = stub.DistributeAveragedGradients(request)
    
    averaged_gradients = b''
    for chunk in averaged_gradients_stream:
        averaged_gradients += chunk.averaged_gradients
        if chunk.is_final_chunk:
            break
    averaged_gradients = deserialize_gradients(averaged_gradients)
    model.set_gradients(averaged_gradients)

def run(worker_id, speed):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = parameter_server_pb2_grpc.ParameterServerStub(channel)
        registration_request = parameter_server_pb2.RegisterWorkerRequest(worker_id=worker_id, worker_capabilities={'speed': str(speed)})
        registration_response = stub.RegisterWorker(registration_request)
        print(f"Worker registration response: {registration_response.message}")

        model_weights = receive_model_weights(stub, worker_id)
        print("Received model weights from parameter server.")
        model, optimizer, criterion = initialize_model(model_weights)

        step_count = 0
        is_end_of_dataset = False
        while not is_end_of_dataset:
            dataset_portion, is_end_of_dataset = fetch_dataset_portion(stub, worker_id)
            print(f"Fetched dataset portion. End of dataset: {is_end_of_dataset}")
            inputs, labels = dataset_portion
            for input, label in zip(dataset_portion[inputs], dataset_portion[labels]):
                # Forward + backward + optimize
                outputs = model(input)
                loss = criterion(outputs, torch.tensor([label], dtype=torch.long))
                loss.backward()
                print(f"Step: {step_count}, Loss: {loss.item()}")  # Print loss for each step

                # Extract gradients and send them to the parameter server
                gradients = model.get_gradients()
                gradients_stream = generate_gradient_chunks(gradients, step_count)
                stub.ReceiveGradients(gradients_stream)

                # Update step_count after processing each batch
                step_count += 1
                receive_and_apply_averaged_gradients(stub, worker_id, step_count, model)
                print(f"Averaged gradients received and applied for step: {step_count}")

                # Step with optimizer
                optimizer.step()

            # Consider adding a condition to break out or reset based on training needs

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(sys.argv)
        print("Usage: python worker.py <worker_id> <worker_speed>")
        sys.exit(1)
    worker_id = sys.argv[1]
    speed = sys.argv[2]
    run(worker_id, speed)