from Models.BaseModel import BaseModule
from Models.GAN import GAN

import torch
import torch.optim as optimizers
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


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


class AutoEncoderTrainer(BaseTrainer):
    def __init__(self, net: BaseModule, train_set, test_set, batch_size=1, use_cuda=True, lr=1e-3):
        super(AutoEncoderTrainer, self).__init__(net, train_set, test_set, batch_size, use_cuda, lr)
        self.preInputTransform = None

    def pre_activation_inputs_modify(self, inputs):
        if self.preInputTransform is not None:
            return self.preInputTransform(inputs)
        return inputs

    def log_test_activations(self, inputs, outputs, targets):
        if not self.test_logged:
            outputs, vec = outputs
            div = self.find_closest_int_divisor(inputs.size(0))

            inputs = self.pre_activation_inputs_modify(inputs)

            if inputs.size(1) == 4:
                inputs = inputs[:, :3, :, :]

            input_grid = torchvision.utils.make_grid(inputs, div, normalize=True, scale_each=True).cpu()
            output_grid = torchvision.utils.make_grid(outputs, div).cpu()
            target_grid = torchvision.utils.make_grid(targets, div).cpu()

            self.writer.add_image('Validation/Input images', input_grid, self.epoch)
            self.writer.add_image('Validation/Output images', output_grid, self.epoch)
            self.writer.add_image('Validation/Target images', target_grid, self.epoch)
        self.test_logged = True

    def log_train_activations(self, inputs, outputs, targets):
        if not self.train_logged:
            outputs, vec = outputs
            div = self.find_closest_int_divisor(inputs.size(0))

            inputs = self.pre_activation_inputs_modify(inputs)

            if inputs.size(1) == 4:
                inputs = inputs[:, :3, :, :]

            input_grid = torchvision.utils.make_grid(inputs, div, normalize=True, scale_each=True).cpu()
            output_grid = torchvision.utils.make_grid(outputs, div).cpu()
            target_grid = torchvision.utils.make_grid(targets, div).cpu()

            self.writer.add_image('Training/Input images', input_grid, self.epoch)
            self.writer.add_image('Training/Output images', output_grid, self.epoch)
            self.writer.add_image('Training/Target images', target_grid, self.epoch)
        self.train_logged = True

    def get_info_from_test_activation(self, inputs, targets):
        with torch.no_grad():
            outputs = self.net(inputs)

            loss_batch = 0
            correct_batch = 0

            outputs, vecs = outputs

            for i in range(0, outputs.shape[0]):

                target = targets[i]
                output = outputs[i]

                loss = self.net.loss(output, target)
                loss_batch += loss.item()
                if loss.item() < self.accuracy_threshold:
                    correct_batch += 1

            return loss_batch, correct_batch, (outputs, vecs)


class FRDTrainer(BaseTrainer):
    def __init__(self, net: BaseModule, train_set, test_set, batch_size=1, use_cuda=True, lr=1e-3):
        super(FRDTrainer, self).__init__(net, train_set, test_set, batch_size, use_cuda, lr)
        self.preInputTransform = None

    def log_train_activations(self, inputs, outputs, targets):
        if not self.train_logged:
            div = self.find_closest_int_divisor(inputs.size(0))

            inputs = self.pre_activation_inputs_modify(inputs)

            if inputs.size(1) == 4:
                inputs = inputs[:, :3, :, :]

            input_grid = torchvision.utils.make_grid(inputs, div, normalize=True, scale_each=True).cpu()
            outputs = outputs.view(-1, 1, 1, 1)
            output_grid = torchvision.utils.make_grid(outputs, div, 1).cpu()

            self.writer.add_image('Training/Input images', input_grid, self.epoch)
            self.writer.add_image('Training/Output images', output_grid, self.epoch)
        self.train_logged = True

    def log_test_activations(self, inputs, outputs, targets):
        if not self.test_logged:
            div = self.find_closest_int_divisor(inputs.size(0))

            inputs = self.pre_activation_inputs_modify(inputs)

            if inputs.size(1) == 4:
                inputs = inputs[:, :3, :, :]

            input_grid = torchvision.utils.make_grid(inputs, div, normalize=True, scale_each=True).cpu()
            outputs = outputs.view(-1, 1, 1, 1)
            output_grid = torchvision.utils.make_grid(outputs, div, 1).cpu()

            self.writer.add_image('Validation/Input images', input_grid, self.epoch)
            self.writer.add_image('Validation/Output images', output_grid, self.epoch)
        self.test_logged = True

    def pre_activation_inputs_modify(self, inputs):
        if self.preInputTransform is not None:
            return self.preInputTransform(inputs)
        return inputs


class GANTrainer(BaseTrainer):
    def __init__(self, net: GAN, train_set, test_set, batch_size=1, use_cuda=True, lr=1e-3):
        super(GANTrainer, self).__init__(net, train_set, test_set, batch_size, use_cuda, lr)
        self.preInputTransform = None

    def pre_activation_inputs_modify(self, inputs):
        if self.preInputTransform is not None:
            return self.preInputTransform(inputs)
        return inputs

    def train_generator(self):
        h = torch.randn(self.batch_size, self.net.generator.vector_size)
        # h.requires_grad = True
        ones = torch.ones(self.batch_size, 1)

        if self.use_cuda:
            h = h.cuda()
            ones = ones.cuda()

        generated_images = self.net(h, "generator")

        self.optimizer[0].zero_grad()
        predictions_fake = self.net(generated_images, "discriminator")
        generator_loss = self.net.loss(predictions_fake, ones)
        generator_loss.backward()
        self.optimizer[0].step()

        return generator_loss

    def train_discriminator(self, inputs):
        h = torch.randn(inputs.size(0), self.net.generator.vector_size)
        ones = torch.ones(inputs.size(0), 1)
        zeros = torch.zeros(inputs.size(0), 1)

        if self.use_cuda:
            inputs = inputs.cuda()
            h = h.cuda()
            ones = ones.cuda()
            zeros = zeros.cuda()

        # inputs.requires_grad = True

        generated_images = self.net(h, "generator").detach()

        self.optimizer[1].zero_grad()
        prediction_real = self.net(inputs, "discriminator")
        discriminator_real_loss = self.net.loss(prediction_real, ones)
        prediction_fake = self.net(generated_images, "discriminator")
        discriminator_fake_loss = self.net.loss(prediction_fake, zeros)
        discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) * 0.5
        discriminator_loss.backward()
        self.optimizer[1].step()

        return discriminator_loss

    def single_traing_iteration(self, inputs, targets):

        generator_loss = self.train_generator()
        discriminator_loss = self.train_discriminator(inputs)

        return generator_loss, discriminator_loss

    def _overfit_first_batch(self, inputs, targets):
        if self.print_messages:
            print("Overfitting first batch...")
        while True:
            loss = self.train_discriminator(inputs)
            loss = loss.item()
            if self.print_messages:
                print(loss)
            if loss < self.overfit_batch_loss:
                break

    def train_iterations(self):
        iteration_loss = 0
        iteration_generator_loss = 0
        iteration_discriminator_loss = 0

        for n, (inputs, _) in enumerate(self.train_loader):
            generator_loss, discriminator_loss = self.single_traing_iteration(inputs, None)

            iteration_generator_loss += generator_loss.item()
            iteration_discriminator_loss += discriminator_loss.item()

            iteration_loss += iteration_generator_loss + iteration_discriminator_loss

            if self.writer is not None:
                self.log_fake_images(inputs)

        return iteration_generator_loss, iteration_discriminator_loss, iteration_loss

    def print_train_loss(self, loss):
        if self.print_messages:
            lr = self.get_lr()
            generator_loss, discriminator_loss, _ = loss
            total_loss = generator_loss + discriminator_loss
            win_ratio = generator_loss / total_loss - discriminator_loss / total_loss
            winner = "none"
            if win_ratio < 0:
                winner = "generator"
            if win_ratio > 0:
                winner = "discriminator"
            score = abs(win_ratio)
            print('[Epoch: %d] generator loss: %.10f, discriminator loss: %.10f, winner: %s (%.5f), LR: %f' %
                  (self.epoch, generator_loss, discriminator_loss, winner, score, lr))

    def create_optimizer(self):
        return (
            optimizers.Adam(self.net.generator.parameters(), lr=self.lr),
            optimizers.SGD(self.net.discriminator.parameters(), lr=self.lr * 4)
        )

    @staticmethod
    def create_scheduler(optimizer):
        generator_optimizer, discriminator_optimizer = optimizer
        return (
            optimizers.lr_scheduler.ReduceLROnPlateau(generator_optimizer, 'min', factor=0.5, eps=0, patience=20),
            optimizers.lr_scheduler.ReduceLROnPlateau(discriminator_optimizer, 'min', factor=0.5, eps=0, patience=20)
        )

    def scheduler_step(self, loss):
        generator_loss, discriminator_loss, _ = loss
        generator_scheduler, discriminator_scheduler = self.scheduler
        # generator_scheduler.step(generator_loss)
        # discriminator_scheduler.step(discriminator_loss)

    def get_lr(self):
        generator_optimizer, discriminator_optimizer = self.optimizer
        generator_lr = generator_optimizer.param_groups[0]["lr"]
        discriminator_lr = discriminator_optimizer.param_groups[0]["lr"]
        return (generator_lr + discriminator_lr) * 0.5

    def get_info_from_test_activation(self, inputs, targets):
        outputs = self.net(inputs, "discriminator")

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

    def log_fake_images(self, inputs):
        if not self.train_logged:
            with torch.no_grad():
                h = torch.randn(inputs.size(0), self.net.generator.vector_size)
                if self.use_cuda:
                    h = h.cuda()
                generated_images = self.net(h, "generator")

                div = self.find_closest_int_divisor(generated_images.size(0))

                fakes_grid = torchvision.utils.make_grid(generated_images, div).cpu()
                inputs_grid = torchvision.utils.make_grid(inputs, div, normalize=True, scale_each=True).cpu()

                self.writer.add_image('Images/Generated', fakes_grid, self.epoch)
                self.writer.add_image('Images/Real', inputs_grid, self.epoch)
        self.train_logged = True

    @staticmethod
    def get_losses(train_loss, test_loss):
        out = {}
        if train_loss is not None:
            out['train'] = train_loss[2]
        if test_loss is not None:
            out['validation'] = test_loss
        return out
