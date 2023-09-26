import os 
from torch.utils.data import Dataset
import torch

from torchvision.io import read_image
import torchvision.transforms as T

class CasiaWebFaceCropIdentification:
    """ Takes casia_web_face_crop_dataset, and generates train, validation, and test datasets for it.
    """
    def __init__(self, directory, n_class, split_ratios, transforms=None, dtype=torch.float32, device='cpu') -> None:
        self.directory = directory
        self.n_class = n_class
        if split_ratios[0] + split_ratios[1] + split_ratios[2] != 1:
            raise ValueError("Incorrect split.")
        self.split_ratios = split_ratios
        self.transforms = transforms
        self.dtype = dtype
        self.device = device
        
        # read the names / classes
        names = torch.tensor([list(map(int, name[:-4].split('_'))) for name in os.listdir(directory)])
        # determine the unique classes
        unique_classes = torch.unique(names[:, 0].flatten())
        # from unique classes and names, determine the class of each sample from 0 to len(unique_classes)
        classes = torch.nonzero(names[:, 0].view(-1, 1) == unique_classes.view(1, -1))[:,1].view(-1, 1)
        # append the class info to the name form the final sample
        self.samples = torch.cat((names, classes), dim=1)
        # now you have [class_name, sample_index, class] for "000{class_name}_00{sample_index}.jpg"
        # ex: 0000045_001.jpg
        # first zeros are appended to fill 7, second ones are appended to fill 3
        
        # make sure you have the correct n_classes by ignoring the last ones.
        split_n_classes = torch.nonzero(self.samples[:,2] == n_class).flatten()[0]
        self.samples = self.samples[:split_n_classes]
        
        # shuffle
        randomizer = torch.randperm(self.samples.shape[0])
        self.samples = self.samples[randomizer]
        
        # split train, validation and test
        total = self.samples.shape[0]
        split_0 = int(split_ratios[0] * total)
        split_1 = int((split_ratios[0]+split_ratios[1]) * total)
        
        self.train_samples = self.samples[:split_0]
        self.validation_samples = self.samples[split_0:split_1]
        self.test_samples = self.samples[split_1:]
        
        # form the datasets
        self.train_dataset = CasiaWebFaceCropIdentificationDataset(directory=directory, 
                                                                   samples=self.train_samples,
                                                                   n_classes=self.n_class,
                                                                   transforms=transforms,
                                                                   dtype=dtype,
                                                                   device=device)
        self.validation_dataset = CasiaWebFaceCropIdentificationDataset(directory=directory, 
                                                                        samples=self.validation_samples, 
                                                                        n_classes=self.n_class,
                                                                        transforms=transforms,
                                                                        dtype=dtype,
                                                                        device=device)
        self.test_dataset = CasiaWebFaceCropIdentificationDataset(directory=directory, 
                                                                  samples=self.test_samples, 
                                                                  n_classes=self.n_class,
                                                                  transforms=transforms,
                                                                  dtype=dtype,
                                                                  device=device)


class CasiaWebFaceCropIdentificationDataset(Dataset):
    def __init__(self, directory, samples, n_classes, transforms=None, dtype=torch.float32, device='cpu') -> None:
        super().__init__()
        
        self.directory = directory
        self.samples = samples
        self.n_classes = n_classes
        self.transforms = transforms
        self.dtype = dtype
        self.device = device
    
    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, index):
        # now you have [class_name, sample_index, class] for "000{class_name}_00{sample_index}.jpg"
        # ex: 0000045_001.jpg
        # first zeros are appended to fill 7, second ones are appended to fill 3
        
        # extract info from samples
        class_name, sample_index, class_number = self.samples[index]
        class_name = str(class_name.item()).zfill(7)
        sample_index = str(sample_index.item()).zfill(3)
        name = class_name + "_" + sample_index + ".jpg"
        
        # read the data
        data = read_image(self.directory + name).to(dtype=self.dtype, device=self.device)
        #T.ToPILImage()(data).show()
        
        # apply transforms
        if self.transforms != None:
            data = self.transforms(data)
            
        #T.ToPILImage()(data).show()
            
        # onehot representation of the class
        label = torch.nn.functional.one_hot(class_number, self.n_classes).to(device=self.device)
        
        return data, label
    

if __name__ == "__main__":
    dtype = torch.float32
    device = 'cpu'
    transform_resize = T.Resize(size=224, antialias=True)
    transform_normalize = T.Normalize(mean=255/2, std=255)
    transforms = torch.nn.Sequential(transform_resize, transform_normalize)
    directory = "datasets/casia_web_face_crop_dataset/"
    n_class = 1000
    split_ratios = (0.90, 0.05, 0.05)
    dataset = CasiaWebFaceCropIdentification(directory, n_class, split_ratios, transforms=transforms, dtype=dtype, device=device)
    
    print(len(dataset.train_dataset))
    print(len(dataset.validation_dataset))
    

# one for verification one for identification. or pass it as parameter