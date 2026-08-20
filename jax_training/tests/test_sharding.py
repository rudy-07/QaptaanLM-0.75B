"""Unit tests for JAX device mesh and sharding configurations."""

import unittest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


class TestSharding(unittest.TestCase):
    """Test device mesh, data sharding, and array distribution."""

    def test_mesh_and_sharding_creation(self):
        devices = jax.devices()
        self.assertGreater(len(devices), 0)

        mesh = Mesh(devices, ("data",))
        data_sharding = NamedSharding(mesh, P("data", None))
        replicated_sharding = NamedSharding(mesh, P())

        # Test putting an array on device with data sharding
        arr = jnp.zeros((len(devices) * 2, 64), dtype=jnp.int32)
        sharded_arr = jax.device_put(arr, data_sharding)

        self.assertEqual(sharded_arr.shape, (len(devices) * 2, 64))
        self.assertEqual(sharded_arr.sharding, data_sharding)


if __name__ == "__main__":
    unittest.main()
