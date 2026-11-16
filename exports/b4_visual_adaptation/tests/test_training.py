"""CPU regression checks for the training objective and resume boundary."""

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import train
from b4.training import LOSS_NAME, answer_loss


class AnswerLossTests(unittest.TestCase):
    def test_matches_full_vocabulary_reference_and_gradient(self):
        for gold in (0, 2, 4):
            with self.subTest(gold=gold):
                logits = torch.tensor([-1.5, 0.3, 0.6, 2.1, -0.7], requires_grad=True)
                reference = logits.detach().clone().requires_grad_(True)
                loss, ce = answer_loss(logits, gold)
                reference_ce = F.cross_entropy(reference[None], torch.tensor([gold]))
                wrong = reference[torch.arange(len(reference)) != gold].max()
                expected = reference_ce + 0.1 * F.relu(0.5 - reference[gold] + wrong)
                torch.testing.assert_close(ce, reference_ce)
                torch.testing.assert_close(loss, expected)
                loss.backward()
                expected.backward()
                torch.testing.assert_close(logits.grad, reference.grad)

    def test_equal_logits_have_finite_loss_and_gradient(self):
        logits = torch.zeros(5, requires_grad=True)
        loss, ce = answer_loss(logits, 2)
        torch.testing.assert_close(ce, torch.tensor(5.0).log())
        torch.testing.assert_close(loss, ce + 0.05)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_positive_scaling_changes_objective(self):
        logits = torch.tensor([-1.5, 0.3, 0.6, 2.1, -0.7])
        loss, _ = answer_loss(logits, 2)
        scaled, _ = answer_loss(logits * 3, 2)
        self.assertGreater(scaled.item(), loss.item())

    def test_confident_answer_has_no_hinge_penalty(self):
        loss, ce = answer_loss(torch.tensor([-2.0, 3.0, 0.1]), 1)
        torch.testing.assert_close(loss, ce)

    def test_bfloat16_inputs_use_float32_loss(self):
        logits = torch.tensor([-1.0, 0.2, 1.5], dtype=torch.bfloat16, requires_grad=True)
        loss, ce = answer_loss(logits, 1)
        expected, _ = answer_loss(logits.detach().float(), 1)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(ce.dtype, torch.float32)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())


class ResumeTests(unittest.TestCase):
    @staticmethod
    def checkpoint(world=2):
        return dict(config=dict(loss=LOSS_NAME, world_size=world), rng=[{} for _ in range(world)])

    def test_accepts_matching_objective_and_gpu_count(self):
        checkpoint = self.checkpoint()
        self.assertIs(train.validate_resume(checkpoint, 2), checkpoint['config'])

    def test_rejects_missing_or_different_objective(self):
        for loss in (None, 'other_objective'):
            with self.subTest(loss=loss):
                checkpoint = self.checkpoint()
                if loss is None:
                    del checkpoint['config']['loss']
                else:
                    checkpoint['config']['loss'] = loss
                with self.assertRaisesRegex(ValueError, 'Checkpoint loss'):
                    train.validate_resume(checkpoint, 2)

    def test_rejects_mismatched_gpu_count_or_rng_layout(self):
        checkpoint = self.checkpoint()
        with self.assertRaisesRegex(ValueError, 'GPU count'):
            train.validate_resume(checkpoint, 1)
        checkpoint['rng'].pop()
        with self.assertRaisesRegex(ValueError, 'RNG state'):
            train.validate_resume(checkpoint, 2)

    def test_incompatible_resume_fails_before_cuda_or_data_access(self):
        checkpoint = self.checkpoint()
        checkpoint['config']['loss'] = 'other_objective'
        argv = ['train.py', '--data', 'unused.jsonl', '--resume', 'checkpoint.pt']
        with patch('sys.argv', argv), patch.object(train.torch, 'load', return_value=checkpoint), \
                patch.object(train.torch.cuda, 'set_device') as cuda, \
                patch.object(train, 'read_rows') as data, patch.object(train, 'Progress') as status:
            with self.assertRaisesRegex(ValueError, 'Checkpoint loss'):
                train.main()
            cuda.assert_not_called()
            data.assert_not_called()
            status.assert_not_called()


if __name__ == '__main__':
    unittest.main()
