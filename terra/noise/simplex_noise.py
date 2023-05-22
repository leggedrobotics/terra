""""
Original source: https://github.com/RoboEden/jux
"""
from typing import List
from typing import NamedTuple

from jax import Array
from jax import lax
from jax import numpy as jnp

INT32_MAX = jnp.iinfo(jnp.int32).max


class SimplexNoise(NamedTuple):
    M_1_PI = 0.31830988618379067154
    F2 = 0.3660254037844386  # 0.5 * (sqrt(3.0) - 1.0)
    G2 = 0.21132486540518713  # (3.0 - sqrt(3.0)) / 6.0
    F3 = 1.0 / 3.0
    G3 = 1.0 / 6.0
    F4 = 0.30901699437494745  # (sqrt(5.0) - 1.0) / 4.0
    G4 = 0.1381966011250105  # (5.0 - sqrt(5.0)) / 20.0
    GRAD3 = jnp.array(
        [
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [1, 0, 1],
            [-1, 0, 1],
            [1, 0, -1],
            [-1, 0, -1],
            [0, 1, 1],
            [0, -1, 1],
            [0, 1, -1],
            [0, -1, -1],
            [1, 0, -1],
            [-1, 0, -1],
            [0, -1, 1],
            [0, 1, 1],
        ]
    )
    GRAD4 = jnp.array(
        [
            [0, 1, 1, 1],
            [0, 1, 1, -1],
            [0, 1, -1, 1],
            [0, 1, -1, -1],
            [0, -1, 1, 1],
            [0, -1, 1, -1],
            [0, -1, -1, 1],
            [0, -1, -1, -1],
            [1, 0, 1, 1],
            [1, 0, 1, -1],
            [1, 0, -1, 1],
            [1, 0, -1, -1],
            [-1, 0, 1, 1],
            [-1, 0, 1, -1],
            [-1, 0, -1, 1],
            [-1, 0, -1, -1],
            [1, 1, 0, 1],
            [1, 1, 0, -1],
            [1, -1, 0, 1],
            [1, -1, 0, -1],
            [-1, 1, 0, 1],
            [-1, 1, 0, -1],
            [-1, -1, 0, 1],
            [-1, -1, 0, -1],
            [1, 1, 1, 0],
            [1, 1, -1, 0],
            [1, -1, 1, 0],
            [1, -1, -1, 0],
            [-1, 1, 1, 0],
            [-1, 1, -1, 0],
            [-1, -1, 1, 0],
            [-1, -1, -1, 0],
        ]
    )
    PERM = jnp.array(
        [
            151,
            160,
            137,
            91,
            90,
            15,
            131,
            13,
            201,
            95,
            96,
            53,
            194,
            233,
            7,
            225,
            140,
            36,
            103,
            30,
            69,
            142,
            8,
            99,
            37,
            240,
            21,
            10,
            23,
            190,
            6,
            148,
            247,
            120,
            234,
            75,
            0,
            26,
            197,
            62,
            94,
            252,
            219,
            203,
            117,
            35,
            11,
            32,
            57,
            177,
            33,
            88,
            237,
            149,
            56,
            87,
            174,
            20,
            125,
            136,
            171,
            168,
            68,
            175,
            74,
            165,
            71,
            134,
            139,
            48,
            27,
            166,
            77,
            146,
            158,
            231,
            83,
            111,
            229,
            122,
            60,
            211,
            133,
            230,
            220,
            105,
            92,
            41,
            55,
            46,
            245,
            40,
            244,
            102,
            143,
            54,
            65,
            25,
            63,
            161,
            1,
            216,
            80,
            73,
            209,
            76,
            132,
            187,
            208,
            89,
            18,
            169,
            200,
            196,
            135,
            130,
            116,
            188,
            159,
            86,
            164,
            100,
            109,
            198,
            173,
            186,
            3,
            64,
            52,
            217,
            226,
            250,
            124,
            123,
            5,
            202,
            38,
            147,
            118,
            126,
            255,
            82,
            85,
            212,
            207,
            206,
            59,
            227,
            47,
            16,
            58,
            17,
            182,
            189,
            28,
            42,
            223,
            183,
            170,
            213,
            119,
            248,
            152,
            2,
            44,
            154,
            163,
            70,
            221,
            153,
            101,
            155,
            167,
            43,
            172,
            9,
            129,
            22,
            39,
            253,
            19,
            98,
            108,
            110,
            79,
            113,
            224,
            232,
            178,
            185,
            112,
            104,
            218,
            246,
            97,
            228,
            251,
            34,
            242,
            193,
            238,
            210,
            144,
            12,
            191,
            179,
            162,
            241,
            81,
            51,
            145,
            235,
            249,
            14,
            239,
            107,
            49,
            192,
            214,
            31,
            181,
            199,
            106,
            157,
            184,
            84,
            204,
            176,
            115,
            121,
            50,
            45,
            127,
            4,
            150,
            254,
            138,
            236,
            205,
            93,
            222,
            114,
            67,
            29,
            24,
            72,
            243,
            141,
            128,
            195,
            78,
            66,
            215,
            61,
            156,
            180,
            151,
            160,
            137,
            91,
            90,
            15,
            131,
            13,
            201,
            95,
            96,
            53,
            194,
            233,
            7,
            225,
            140,
            36,
            103,
            30,
            69,
            142,
            8,
            99,
            37,
            240,
            21,
            10,
            23,
            190,
            6,
            148,
            247,
            120,
            234,
            75,
            0,
            26,
            197,
            62,
            94,
            252,
            219,
            203,
            117,
            35,
            11,
            32,
            57,
            177,
            33,
            88,
            237,
            149,
            56,
            87,
            174,
            20,
            125,
            136,
            171,
            168,
            68,
            175,
            74,
            165,
            71,
            134,
            139,
            48,
            27,
            166,
            77,
            146,
            158,
            231,
            83,
            111,
            229,
            122,
            60,
            211,
            133,
            230,
            220,
            105,
            92,
            41,
            55,
            46,
            245,
            40,
            244,
            102,
            143,
            54,
            65,
            25,
            63,
            161,
            1,
            216,
            80,
            73,
            209,
            76,
            132,
            187,
            208,
            89,
            18,
            169,
            200,
            196,
            135,
            130,
            116,
            188,
            159,
            86,
            164,
            100,
            109,
            198,
            173,
            186,
            3,
            64,
            52,
            217,
            226,
            250,
            124,
            123,
            5,
            202,
            38,
            147,
            118,
            126,
            255,
            82,
            85,
            212,
            207,
            206,
            59,
            227,
            47,
            16,
            58,
            17,
            182,
            189,
            28,
            42,
            223,
            183,
            170,
            213,
            119,
            248,
            152,
            2,
            44,
            154,
            163,
            70,
            221,
            153,
            101,
            155,
            167,
            43,
            172,
            9,
            129,
            22,
            39,
            253,
            19,
            98,
            108,
            110,
            79,
            113,
            224,
            232,
            178,
            185,
            112,
            104,
            218,
            246,
            97,
            228,
            251,
            34,
            242,
            193,
            238,
            210,
            144,
            12,
            191,
            179,
            162,
            241,
            81,
            51,
            145,
            235,
            249,
            14,
            239,
            107,
            49,
            192,
            214,
            31,
            181,
            199,
            106,
            157,
            184,
            84,
            204,
            176,
            115,
            121,
            50,
            45,
            127,
            4,
            150,
            254,
            138,
            236,
            205,
            93,
            222,
            114,
            67,
            29,
            24,
            72,
            243,
            141,
            128,
            195,
            78,
            66,
            215,
            61,
            156,
            180,
        ]
    )
    SIMPLEX = jnp.array(
        [
            [0, 1, 2, 3],
            [0, 1, 3, 2],
            [0, 0, 0, 0],
            [0, 2, 3, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 2, 3, 0],
            [0, 2, 1, 3],
            [0, 0, 0, 0],
            [0, 3, 1, 2],
            [0, 3, 2, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 3, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 2, 0, 3],
            [0, 0, 0, 0],
            [1, 3, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 3, 0, 1],
            [2, 3, 1, 0],
            [1, 0, 2, 3],
            [1, 0, 3, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 3, 1],
            [0, 0, 0, 0],
            [2, 1, 3, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 1, 3],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [3, 0, 1, 2],
            [3, 0, 2, 1],
            [0, 0, 0, 0],
            [3, 1, 2, 0],
            [2, 1, 0, 3],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [3, 1, 0, 2],
            [0, 0, 0, 0],
            [3, 2, 0, 1],
            [3, 2, 1, 0],
        ]
    )

    @classmethod
    def dispatch_noise2(
        cls,
        x: Array,
        y: Array,
        octaves: jnp.int32 = 1,
        persistence: jnp.float32 = 0.5,
        lacunarity: jnp.float32 = 2.0,
        repeatx: jnp.int32 = INT32_MAX,
        repeaty: jnp.int32 = INT32_MAX,
        z: jnp.float32 = 0.0,
    ):
        no_repeat = (repeatx == INT32_MAX) and (repeaty == INT32_MAX)
        tiled_repeaty = (repeatx == INT32_MAX) and (repeaty != INT32_MAX)
        tiled_repeatx = (repeatx != INT32_MAX) and (repeaty == INT32_MAX)
        tiled_repeatxy = (repeatx != INT32_MAX) and (repeaty != INT32_MAX)

        def _no_repeat(x, y, octaves, persistence, lacunarity, repeatx, repeaty, z):
            noise = cls.fbm_noise2(x, y, z, octaves, persistence, lacunarity)
            return noise

        def _tiled_repeaty(x, y, octaves, persistence, lacunarity, repeatx, repeaty, z):
            w = z
            yf = y * 2.0 / repeaty
            yr = repeaty * cls.M_1_PI * 0.5
            vy = jnp.sin(yf)
            vyz = jnp.cos(yf)
            y = vy * yr
            w = w + vyz * yr
            noise = cls.fbm_noise3(x, y, w, octaves, persistence, lacunarity)
            return noise

        def _tiled_repeatx(x, y, octaves, persistence, lacunarity, repeatx, repeaty, z):
            xf = x * 2.0 / repeatx
            xr = repeatx * cls.M_1_PI * 0.5
            vx = jnp.sin(xf)
            vxz = jnp.cos(xf)
            x = vx * xr
            z = z + vxz * xr
            noise = cls.fbm_noise3(x, y, z, octaves, persistence, lacunarity)
            return noise

        def _tiled_repeatxy(
            x, y, octaves, persistence, lacunarity, repeatx, repeaty, z
        ):
            w = z
            yf = y * 2.0 / repeaty
            yr = repeaty * cls.M_1_PI * 0.5
            vy = jnp.sin(yf)
            vyz = jnp.cos(yf)
            y = vy * yr
            w = w + vyz * yr

            xf = x * 2.0 / repeatx
            xr = repeatx * cls.M_1_PI * 0.5
            vx = jnp.sin(xf)
            vxz = jnp.cos(xf)
            x = vx * xr
            z = z + vxz * xr
            noise = cls.fbm_noise4(x, y, z, w, octaves, persistence, lacunarity)
            return noise

        no_repeat * 1 + tiled_repeaty * 2 + tiled_repeatx * 3 + tiled_repeatxy * 4 - 1
        # jax.debug.print("condition: {condition}", condition=condition)

        # return lax.switch(condition, [_no_repeat, _tiled_repeaty, _tiled_repeatx, _tiled_repeatxy],
        #                   *(x, y, octaves, persistence, lacunarity, repeatx, repeaty, z))
        noise = _no_repeat(x, y, octaves, persistence, lacunarity, repeatx, repeaty, z)
        # jax.debug.print("noise_jux: {noise}", noise=noise)
        return noise

    @classmethod
    def fbm_noise2(cls, x, y, z, octaves, persistence, lacunarity):
        freq = 1.0
        amp = 1.0
        max = 1.0
        total = cls.noise2(x + z, y + z)

        def body_func(i, val):
            freq, amp, max, total = val
            freq = freq * lacunarity
            amp = amp * persistence
            max = max + amp
            total = total + cls.noise2(x * freq + z, y * freq + z) * amp
            return freq, amp, max, total

        freq, amp, max, total = lax.fori_loop(
            lower=1, upper=octaves, body_fun=body_func, init_val=(freq, amp, max, total)
        )
        return total / max

    @classmethod
    def fbm_noise3(cls, x, y, z, octaves, persistence, lacunarity):
        freq = 1.0
        amp = 1.0
        max = 1.0
        total = cls.noise3(x, y, z)

        def body_func(i, val):
            freq, amp, max, total = val
            freq = freq * lacunarity
            amp = amp * persistence
            max = max + amp
            total = total + cls.noise3(x * freq, y * freq, z * freq) * amp
            return freq, amp, max, total

        freq, amp, max, total = lax.fori_loop(
            lower=1, upper=octaves, body_fun=body_func, init_val=(freq, amp, max, total)
        )
        return total / max

    @classmethod
    def fbm_noise4(cls, x, y, z, w, octaves, persistence, lacunarity):
        freq = 1.0
        amp = 1.0
        max = 1.0
        total = cls.noise4(x, y, z, w)

        def body_func(i, val):
            freq, amp, max, total = val
            freq = freq * lacunarity
            amp = amp * persistence
            max = max + amp
            total = total + cls.noise4(x * freq, y * freq, z * freq, w * freq) * amp
            return freq, amp, max, total

        freq, amp, max, total = lax.fori_loop(
            lower=1, upper=octaves, body_fun=body_func, init_val=(freq, amp, max, total)
        )
        return total / max

    @classmethod
    def noise2(cls, x: Array, y: Array):
        s = (x + y) * cls.F2
        i = jnp.floor(x + s)
        j = jnp.floor(y + s)
        t = (i + j) * cls.G2

        xx: list[Array] = [None] * 3
        yy: list[Array] = [None] * 3
        f: list[Array] = [None] * 3
        noise: list[Array] = [jnp.zeros_like(x)] * 3
        g: list[Array] = [None] * 3

        xx[0] = x - (i - t)
        yy[0] = y - (j - t)

        i1 = xx[0] > yy[0]
        j1 = xx[0] <= yy[0]

        xx[2] = xx[0] + cls.G2 * 2.0 - 1.0
        yy[2] = yy[0] + cls.G2 * 2.0 - 1.0
        xx[1] = xx[0] - i1 + cls.G2
        yy[1] = yy[0] - j1 + cls.G2

        I = jnp.int32(i) & 255
        J = jnp.int32(j) & 255

        g[0] = cls.PERM[I + cls.PERM[J]] % 12
        g[1] = cls.PERM[I + i1 + cls.PERM[J + j1]] % 12
        g[2] = cls.PERM[I + 1 + cls.PERM[J + 1]] % 12

        for c in range(3):
            f[c] = 0.5 - xx[c] * xx[c] - yy[c] * yy[c]

        for c in range(3):
            noise[c] = noise[c] + (
                f[c] ** 4
                * (cls.GRAD3[g[c]][..., 0] * xx[c] + cls.GRAD3[g[c]][..., 1] * yy[c])
            ) * (f[c] > 0)
        # jax.debug.print("{noise}", noise=noise)
        return (noise[0] + noise[1] + noise[2]) * 70.0

    @classmethod
    def noise3(cls, x, y, z):
        f: list[Array] = [jnp.zeros_like(x)] * 4
        g: list[Array] = [None] * 4
        noise: list[Array] = [jnp.zeros_like(x)] * 4
        o1: list[Array] = [jnp.zeros_like(x, dtype=jnp.bool_)] * 4
        o2: list[Array] = [jnp.zeros_like(x, dtype=jnp.bool_)] * 4
        s = (x + y + z) * cls.F3
        i = jnp.floor(x + s)
        j = jnp.floor(y + s)
        k = jnp.floor(z + s)
        t = (i + j + k) * cls.G3

        pos: list[list[Array]] = [[None] * 3 for i in range(4)]
        pos[0][0] = x - (i - t)
        pos[0][1] = y - (j - t)
        pos[0][2] = z - (k - t)

        def _assign(o, value, condition):
            for i in range(3):
                o[i] = o[i] | jnp.bool_(value[i] * condition)
            return o

        cond1 = pos[0][0] >= pos[0][1]
        cond2 = ~cond1
        cond11 = cond1 & (pos[0][1] >= pos[0][2])
        cond12 = cond1 & (pos[0][0] >= pos[0][2])
        cond13 = cond1 & ~(cond11 | cond2)

        cond21 = cond2 & (pos[0][1] < pos[0][2])
        cond22 = cond2 & (pos[0][0] < pos[0][2])
        cond23 = cond2 & ~(cond21 | cond22)
        o1 = _assign(o1, jnp.array([1, 0, 0]), cond11)
        o2 = _assign(o2, jnp.array([1, 1, 0]), cond11)

        o1 = _assign(o1, jnp.array([1, 0, 0]), cond12)
        o2 = _assign(o2, jnp.array([1, 0, 1]), cond12)

        o1 = _assign(o1, jnp.array([0, 0, 1]), cond13)
        o2 = _assign(o2, jnp.array([1, 0, 1]), cond13)

        o1 = _assign(o1, jnp.array([0, 0, 1]), cond21)
        o2 = _assign(o2, jnp.array([0, 1, 1]), cond21)

        o1 = _assign(o1, jnp.array([0, 1, 0]), cond22)
        o2 = _assign(o2, jnp.array([0, 1, 1]), cond22)

        o1 = _assign(o1, jnp.array([0, 1, 0]), cond23)
        o2 = _assign(o2, jnp.array([1, 1, 0]), cond23)

        for c in range(3):
            pos[3][c] = pos[0][c] - 1.0 + 3.0 * cls.G3
            pos[2][c] = pos[0][c] - o2[c] + 2.0 * cls.G3
            pos[1][c] = pos[0][c] - o1[c] + cls.G3

        I = jnp.int32(i) & 255
        J = jnp.int32(j) & 255
        K = jnp.int32(k) & 255
        g[0] = cls.PERM[I + cls.PERM[J + cls.PERM[K]]] % 12
        g[1] = cls.PERM[I + o1[0] + cls.PERM[J + o1[1] + cls.PERM[o1[2] + K]]] % 12
        g[2] = cls.PERM[I + o2[0] + cls.PERM[J + o2[1] + cls.PERM[o2[2] + K]]] % 12
        g[3] = cls.PERM[I + 1 + cls.PERM[J + 1 + cls.PERM[K + 1]]] % 12

        for c in range(3):
            f[c] = (
                0.6
                - pos[c][0] * pos[c][0]
                - pos[c][1] * pos[c][1]
                - pos[c][2] * pos[c][2]
            )

        dot3 = lambda x, y: (x[0] * y[..., 0] + x[1] * y[..., 1] + x[2] * y[..., 2])
        for c in range(3):
            noise[c] = (f[c] * f[c] * f[c] * f[c] * dot3(pos[c], cls.GRAD3[g[c]])) * (
                f[c] > 0
            )

        return (noise[0] + noise[1] + noise[2] + noise[3]) * 32.0

    @classmethod
    def noise4(cls, x: Array, y: Array, z: Array, w: Array):
        noise: list[Array] = [jnp.zeros_like(x)] * 5

        s = (x + y + z + w) * cls.F4
        i = jnp.floor(x + s).astype(jnp.int32)
        j = jnp.floor(y + s).astype(jnp.int32)
        k = jnp.floor(z + s).astype(jnp.int32)
        l = jnp.floor(w + s).astype(jnp.int32)
        t = (i + j + k + l) * cls.G4

        x0 = x - (i - t)
        y0 = y - (j - t)
        z0 = z - (k - t)
        w0 = w - (l - t)

        c = (
            (x0 > y0) * 32
            + (x0 > z0) * 16
            + (y0 > z0) * 8
            + (x0 > w0) * 4
            + (y0 > w0) * 2
            + (z0 > w0)
        )
        i1 = cls.SIMPLEX[c][..., 0] >= 3
        j1 = cls.SIMPLEX[c][..., 1] >= 3
        k1 = cls.SIMPLEX[c][..., 2] >= 3
        l1 = cls.SIMPLEX[c][..., 3] >= 3
        i2 = cls.SIMPLEX[c][..., 0] >= 2
        j2 = cls.SIMPLEX[c][..., 1] >= 2
        k2 = cls.SIMPLEX[c][..., 2] >= 2
        l2 = cls.SIMPLEX[c][..., 3] >= 2
        i3 = cls.SIMPLEX[c][..., 0] >= 1
        j3 = cls.SIMPLEX[c][..., 1] >= 1
        k3 = cls.SIMPLEX[c][..., 2] >= 1
        l3 = cls.SIMPLEX[c][..., 3] >= 1

        x1 = x0 - i1 + cls.G4
        y1 = y0 - j1 + cls.G4
        z1 = z0 - k1 + cls.G4
        w1 = w0 - l1 + cls.G4
        x2 = x0 - i2 + 2.0 * cls.G4
        y2 = y0 - j2 + 2.0 * cls.G4
        z2 = z0 - k2 + 2.0 * cls.G4
        w2 = w0 - l2 + 2.0 * cls.G4
        x3 = x0 - i3 + 3.0 * cls.G4
        y3 = y0 - j3 + 3.0 * cls.G4
        z3 = z0 - k3 + 3.0 * cls.G4
        w3 = w0 - l3 + 3.0 * cls.G4
        x4 = x0 - 1.0 + 4.0 * cls.G4
        y4 = y0 - 1.0 + 4.0 * cls.G4
        z4 = z0 - 1.0 + 4.0 * cls.G4
        w4 = w0 - 1.0 + 4.0 * cls.G4

        I = i & 255
        J = j & 255
        K = k & 255
        L = l & 255
        gi0 = cls.PERM[I + cls.PERM[J + cls.PERM[K + cls.PERM[L]]]] & 0x1F
        gi1 = (
            cls.PERM[I + i1 + cls.PERM[J + j1 + cls.PERM[K + k1 + cls.PERM[L + l1]]]]
            & 0x1F
        )
        gi2 = (
            cls.PERM[I + i2 + cls.PERM[J + j2 + cls.PERM[K + k2 + cls.PERM[L + l2]]]]
            & 0x1F
        )
        gi3 = (
            cls.PERM[I + i3 + cls.PERM[J + j3 + cls.PERM[K + k3 + cls.PERM[L + l3]]]]
            & 0x1F
        )
        gi4 = (
            cls.PERM[I + 1 + cls.PERM[J + 1 + cls.PERM[K + 1 + cls.PERM[L + 1]]]] & 0x1F
        )

        dot4 = lambda v, x, y, z, w: (
            v[..., 0] * x + v[..., 1] * y + v[..., 3] * z + v[..., 4] * w
        )
        t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0 - w0 * w0
        noise[0] = t0**4 * dot4(cls.GRAD4[gi0], x0, y0, z0, w0) * (t0 > 0)
        t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1 - w1 * w1
        noise[1] = t1**4 * dot4(cls.GRAD4[gi1], x1, y1, z1, w1) * (t1 > 0)
        t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2 - w2 * w2
        noise[2] = t2**4 * dot4(cls.GRAD4[gi2], x2, y2, z2, w2) * (t2 > 0)
        t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3 - w3 * w3
        noise[3] = t3**4 * dot4(cls.GRAD4[gi3], x3, y3, z3, w3) * (t3 > 0)
        t4 = 0.6 - x4 * x4 - y4 * y4 - z4 * z4 - w4 * w4
        noise[4] = t4**4 * dot4(cls.GRAD4[gi4], x4, y4, z4, w4) * (t4 > 0)

        return 27.0 * (noise[0] + noise[1] + noise[2] + noise[3] + noise[4])
