import grpc
from concurrent import futures
from threading import Lock

import torch
import parameter_server_pb2_grpc, parameter_server_pb2
from model import BasicCNN
from serializers import serialize_model_weights, deserialize_gradients, serialize_gradients
from datasets import load_dataset
import pickle

class ParameterServer(parameter_server_pb2_grpc.ParameterServerServicer):
    def __init__(self):
        self.workers = {}
        self.global_step = 0
        self.model = BasicCNN()
        self.dataset = load_dataset("mnist")
        self.dataset_position = 0
        self.dataset_chunksize = 1000
        self.averaged_gradients = None
        self.lock = Lock()
        self.lock2 = Lock()

    # No change needed here, you already log worker registration
    def RegisterWorker(self, request, context):
        worker_id = request.worker_id
        self.workers[worker_id] = {
            'capabilities': request.worker_capabilities,
            'current_step': self.global_step,
        }
        print(f"Worker {worker_id} registered with capabilities: {request.worker_capabilities}")
        return parameter_server_pb2.RegisterWorkerResponse(success=True, message="Registered successfully")

    # Log when starting to distribute model weights
    def DistributeModelWeights(self, request_iterator, context):
        for request in request_iterator:
            worker_id = request.worker_id
            print(f"Starting to distribute model weights to worker {worker_id}")
            model_weights = serialize_model_weights(self.model)
            for chunk in self._generate_weight_chunks(model_weights):
                yield parameter_server_pb2.ModelWeightsResponse(
                    model_weights_chunk=chunk, 
                    is_final_chunk=False
                )
            yield parameter_server_pb2.ModelWeightsResponse(
                model_weights_chunk=b'', 
                is_final_chunk=True
            )

    # Log when a dataset portion has been sent
    def DistributeDataset(self, request, context):
        worker_id = request.worker_id
        print(f"Distributing dataset portion to worker {worker_id}")
        dataset_portion, is_end_of_dataset = self._get_dataset_portion()
        return parameter_server_pb2.DatasetResponse(
            dataset_portion=pickle.dumps(dataset_portion),
            is_end_of_dataset=is_end_of_dataset
        )

    def ReceiveGradients(self, request_iterator, context):
            gradients_chunks = []
            worker_id = None
            training_step = None
            for request in request_iterator:
                worker_id = request.worker_id
                training_step = request.training_step
                gradients_chunks.append(request.gradients)
            
            combined_gradients_bytes = b''.join(gradients_chunks)
            gradients = deserialize_gradients(combined_gradients_bytes)
            
            # Use the lock to ensure atomicity of the check-update-clear operation
            with self.lock:
                if worker_id is not None and training_step is not None:
                    self.workers[worker_id]['gradients'] = gradients
                    self.workers[worker_id]['step'] = training_step

                    if all(worker.get('gradients') is not None for worker in self.workers.values()):
                        averaged_gradients = self._average_gradients()
                        self.model.set_gradients(averaged_gradients)
                        self.global_step += 1
                        for worker in self.workers.values():
                            worker['gradients'] = None  # Clear gradients after they've been averaged
                        print(f"Gradients received and processed from worker {worker_id} at step {training_step}")
                        self.averaged_gradients = averaged_gradients
                        return parameter_server_pb2.GradientsResponse(success=True)
                    else:
                        return parameter_server_pb2.GradientsResponse(success=True)

    # Log when starting to distribute averaged gradients
    def DistributeAveragedGradients(self, request, context):
        training_step = request.training_step
        print(f"Starting to distribute averaged gradients for training step {training_step}")
        serialized_gradients = serialize_gradients(self.averaged_gradients)
        
        for chunk in self._generate_gradient_chunks(serialized_gradients):
            yield parameter_server_pb2.AveragedGradientsResponse(
                averaged_gradients=chunk, 
                is_final_chunk=False
            )
        yield parameter_server_pb2.AveragedGradientsResponse(
            averaged_gradients=b'', 
            is_final_chunk=True
        )

    def _generate_weight_chunks(self, model_weights, chunk_size=1024 * 1024):
        """Yield model weight chunks for streaming."""
        for i in range(0, len(model_weights), chunk_size):

            yield model_weights[i:i+chunk_size]
    def _generate_weight_chunks(self, model_weights, chunk_size=1024 * 1024):
        for i in range(0, len(model_weights), chunk_size):
            yield model_weights[i:i+chunk_size]
            

    def _generate_gradient_chunks(self, gradients, chunk_size=1024 * 1024):
        """Yield gradient chunks for streaming."""
        for i in range(0, len(gradients), chunk_size):
            yield gradients[i:i+chunk_size]

    def _average_gradients(self):
        averaged_gradients = {}
        for key in self.workers[next(iter(self.workers))]['gradients'].keys():
            # Gather all gradients for the current key from all workers
            grads = [self.workers[worker_id]['gradients'][key] for worker_id in self.workers if self.workers[worker_id]['gradients']]
            
            # Stack the gradients along a new dimension (at the 0th dimension), resulting in a new tensor
            stacked_grads = torch.stack(grads, dim=0)
            
            # Calculate the mean across the 0th dimension, effectively averaging the gradients
            averaged_gradients[key] = torch.mean(stacked_grads, dim=0)
        return averaged_gradients

    def _get_dataset_portion(self):
        with self.lock:
            dataset_split = 'train'
            end_position = self.dataset_position + self.dataset_chunksize
            if end_position >= len(self.dataset[dataset_split]):
                dataset_portion = self.dataset[dataset_split][self.dataset_position:]
                self.dataset_position = 0  # Reset for simplicity; adjust based on your logic
                is_end_of_dataset = True
            else:
                dataset_portion = self.dataset[dataset_split][self.dataset_position:end_position]
                self.dataset_position += self.dataset_chunksize
                is_end_of_dataset = False
            return dataset_portion, is_end_of_dataset

    def _get_next_dataset_portion(self, worker_id, split='train'):
            worker_info = self.workers[worker_id]
            start_pos = worker_info['dataset_position']
            end_pos = start_pos + self.dataset_chunksize

            if end_pos >= len(self.dataset[split]):
                dataset_portion = self.dataset[split][start_pos:]
                worker_info['dataset_position'] = len(self.dataset[split])  
                is_end_of_dataset = True
            else:
                dataset_portion = self.dataset[split][start_pos:end_pos]
                worker_info['dataset_position'] = end_pos  
                is_end_of_dataset = False

            return dataset_portion, is_end_of_dataset

        
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    parameter_server_pb2_grpc.add_ParameterServerServicer_to_server(ParameterServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()