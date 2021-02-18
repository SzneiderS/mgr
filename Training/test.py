from Models.BaseModel import BaseModule
import torch
import torchvision.transforms as transforms
import torch.optim as optimizers


class BaseTrainer:
    def __init__(self, net: BaseModule or None, train_set, test_set, batch_size=1, use_cuda=True, lr=1e-3):
        if net is not None:
            self.net = net

        self.print_messages = True

        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.writer = None

        self.toTensor = transforms.ToTensor()

        self.saveEveryNthEpoch = 15
        self.saveName = None
        self.loadName = None

        self.train_set = train_set
        self.test_set = test_set

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True,
                                                        num_workers=4)
        if test_set is not None:
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=False,
                                                           shuffle=False, num_workers=4)

        self.accuracy_threshold = 0.01

        self.accuracy_stop = 0.97

        self.epoch = 1

        self.batch_size = batch_size

        self.train_logged = False
        self.test_logged = False

        self.optimizer = None
        self.scheduler = None

        self.after_epoch_func = None

        self.force_full_load = False

        self.lr = lr

        self.overfit_batch_loss = None

    def load_network(self):
        if self.loadName is not None:
            if self.net is None or self.force_full_load:
                self.net = BaseModule.load(self.loadName)
                if self.print_messages:
                    print("Network loaded")
            else:
                if self.net.load_state(self.loadName) and self.print_messages:
                    print("Network loaded")

    def create_optimizer(self):
        return optimizers.Adam(self.net.parameters(), lr=self.lr)

    @staticmethod
    def create_scheduler(optimizer):
        return optimizers.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, eps=0, patience=5)

    @staticmethod
    def get_losses(train_loss, test_loss):
        out = {}
        if train_loss is not None:
            out['train'] = train_loss
        if test_loss is not None:
            out['validation'] = test_loss
        return out

    def _overfit_first_batch(self, inputs, targets):
        if self.print_messages:
            print("Overfitting first batch...")
        while True:
            loss, _ = self.single_traing_iteration(inputs, targets)
            loss = loss.item()
            if self.print_messages:
                print(loss)
            if loss < self.overfit_batch_loss:
                break

    def run(self):
        self.load_network()

        if self.use_cuda:
            self.net = self.net.cuda()

        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler(self.optimizer)

        test_loss = None

        if len(self.train_set) > 0:
            if self.overfit_batch_loss is not None:
                for (inputs, targets) in self.train_loader:
                    self.net.train(True)
                    self._overfit_first_batch(inputs, targets)
                    self.net.train(False)
                    break
            while True:
                accuracy = None
                self.train_logged = False
                train_loss = self.train()
                if self.test_set is not None and len(self.test_set) > 0:
                    self.test_logged = False
                    test_loss, _, accuracy = self.test()
                self.epoch += 1

                if self.writer is not None:
                    scalars = self.get_losses(train_loss, test_loss)
                    if bool(scalars):
                        self.writer.add_scalars('Loss', scalars, self.epoch)

                if self.saveName is not None and self.epoch % self.saveEveryNthEpoch == 0:
                    self.net.save(self.saveName)
                    if self.print_messages:
                        print("Network saved")

                if accuracy is not None and accuracy > self.accuracy_stop:
                    break

                if self.after_epoch_func:
                    self.after_epoch_func(self.epoch)

            if self.saveName is not None:
                self.net.save(self.saveName)
                if self.print_messages:
                    print("Network saved")

    @staticmethod
    def find_closest_int_divisor(size, maximum=6):
        square_root = size ** 0.5
        if square_root == int(square_root):
            if maximum is None:
                return int(square_root)
            return int(min(square_root, maximum))
        floored = int(square_root)
        change = 0
        while True:
            if (floored - change) > 1 and (size % (floored - change)) == 0:
                divisor = floored - change
                break
            if (size % (floored + change)) == 0:
                divisor = floored + change
                break
            change += 1
        if maximum is None:
            return int(divisor)
        return int(min(divisor, maximum))

    def log_train_activations(self, inputs, outputs, targets):
        pass

    def pre_activation_inputs_modify(self, inputs):
        return inputs

    def single_traing_iteration(self, inputs, targets):
        self.optimizer.zero_grad()

        if self.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = self.net(self.pre_activation_inputs_modify(inputs))

        # inputs.requires_grad = True

        loss = self.net.loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

        return loss, outputs

    def train_iterations(self):
        iteration_loss = 0
        for n, (inputs, targets) in enumerate(self.train_loader):
            loss, outputs = self.single_traing_iteration(inputs, targets)
            iteration_loss += loss.item()

            if self.writer is not None:
                self.log_train_activations(inputs, outputs, targets)

        return iteration_loss

    def scheduler_step(self, loss):
        self.scheduler.step(loss)

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def print_train_loss(self, loss):
        if self.print_messages:
            lr = self.get_lr()
            print('[Epoch: %d] loss: %.10f, LR: %f' % (self.epoch, loss, lr))

    def train(self):
        self.net.train(True)
        total_loss = self.train_iterations()

        self.scheduler_step(total_loss)

        self.print_train_loss(total_loss)

        return total_loss

    def get_info_from_test_activation(self, inputs, targets):
        outputs = self.net(inputs)

        loss_batch = 0
        correct_batch = 0

        for i in range(0, outputs.shape[0]):

            target = targets[i]
            output = outputs[i]

            loss = self.net.loss(output, target)
            loss_batch += loss.item()
            if loss.item() < self.accuracy_threshold:
                correct_batch += 1

        return loss_batch, correct_batch, outputs

    def log_test_activations(self, inputs, outputs, targets):
        pass

    def test(self):
        self.net.train(False)
        correct = 0
        total = len(self.test_set)
        total_loss = 0
        with torch.no_grad():
            for (inputs, targets) in self.test_loader:

                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                loss_batch, correct_batch, outputs = self.get_info_from_test_activation(
                    self.pre_activation_inputs_modify(inputs), targets)

                total_loss += loss_batch
                correct += correct_batch

                if self.writer is not None:
                    self.log_test_activations(inputs, outputs, targets)

            accuracy = float(correct / total)

            if self.print_messages:
                print('Accuracy of the network on the test images: %.3f %%, loss: %f' % (100 * accuracy, total_loss))

        return total_loss, correct, accuracy