// Benchmark results transcribed directly from the repository README.
window.GAZEWORLD_RESULTS = {
  "supervised": {
    "columns": [
      "CheXpert",
      "RSNA",
      "SIIM-ACR"
    ],
    "rows": [
      {
        "name": "BioViL",
        "ours": false,
        "values": [
          [
            54.19,
            60.37,
            66.55
          ],
          [
            70.74,
            72.84,
            78.48
          ],
          [
            62.98,
            68.39,
            78.2
          ]
        ]
      },
      {
        "name": "MedKLIP",
        "ours": false,
        "values": [
          [
            null,
            null,
            79.43
          ],
          [
            78.62,
            82.76,
            84.67
          ],
          [
            74.5,
            78.27,
            84.67
          ]
        ]
      },
      {
        "name": "MGCA",
        "ours": false,
        "values": [
          [
            67.12,
            76.1,
            81.57
          ],
          [
            80.41,
            85.67,
            88.24
          ],
          [
            86.14,
            88.13,
            91.84
          ]
        ]
      },
      {
        "name": "GLoRIA",
        "ours": false,
        "values": [
          [
            67.4,
            77.37,
            81.74
          ],
          [
            81.9,
            81.24,
            88.57
          ],
          [
            null,
            null,
            null
          ]
        ]
      },
      {
        "name": "PRIOR",
        "ours": false,
        "values": [
          [
            65.51,
            70.7,
            79.11
          ],
          [
            79.74,
            79.72,
            83.95
          ],
          [
            81.96,
            82.76,
            85.33
          ]
        ]
      },
      {
        "name": "CheXWorld",
        "ours": false,
        "values": [
          [
            62.81,
            80.03,
            85.39
          ],
          [
            78.3,
            84.12,
            88.66
          ],
          [
            62.31,
            81.54,
            91.34
          ]
        ]
      },
      {
        "name": "MedCLIP",
        "ours": false,
        "values": [
          [
            74.65,
            82.88,
            85.58
          ],
          [
            82.31,
            84.16,
            87.86
          ],
          [
            86.09,
            87.41,
            91.46
          ]
        ]
      },
      {
        "name": "EGMA",
        "ours": false,
        "values": [
          [
            71.21,
            79.79,
            85.45
          ],
          [
            82.49,
            84.73,
            88.09
          ],
          [
            87.17,
            87.91,
            88.98
          ]
        ]
      },
      {
        "name": "GazeWorld (ours)",
        "ours": true,
        "values": [
          [
            78.37,
            83.41,
            87.61
          ],
          [
            83.27,
            85.75,
            90.15
          ],
          [
            87.85,
            90.48,
            94.27
          ]
        ]
      }
    ]
  },
  "zero": {
    "columns": [
      "CheXpert 5×200",
      "RSNA",
      "SIIM-ACR"
    ],
    "rows": [
      {
        "name": "CLIP",
        "ours": false,
        "values": [
          [
            15.5,
            19.6,
            10.99
          ],
          [
            21.67,
            23.46,
            19.87
          ],
          [
            42.13,
            46.11,
            40.13
          ]
        ]
      },
      {
        "name": "GLoRIA",
        "ours": false,
        "values": [
          [
            49.18,
            51.1,
            47.25
          ],
          [
            60.77,
            61.53,
            60.09
          ],
          [
            null,
            null,
            null
          ]
        ]
      },
      {
        "name": "MGCA",
        "ours": false,
        "values": [
          [
            42.53,
            43.39,
            39.67
          ],
          [
            59.19,
            60.82,
            59.57
          ],
          [
            36.6,
            38.45,
            34.95
          ]
        ]
      },
      {
        "name": "MedCLIP",
        "ours": false,
        "values": [
          [
            56.63,
            57.4,
            55.72
          ],
          [
            58.08,
            62.17,
            40.06
          ],
          [
            56.17,
            56.89,
            55.44
          ]
        ]
      },
      {
        "name": "EGMA",
        "ours": false,
        "values": [
          [
            56.09,
            57.15,
            55.03
          ],
          [
            57.46,
            66.85,
            43.08
          ],
          [
            59.79,
            63.24,
            61.04
          ]
        ]
      },
      {
        "name": "GazeWorld (ours)",
        "ours": true,
        "values": [
          [
            58.13,
            59.42,
            56.85
          ],
          [
            62.84,
            70.18,
            47.31
          ],
          [
            64.29,
            66.59,
            63.26
          ]
        ]
      }
    ]
  },
  "scanpath": {
    "columns": [
      "SM ↑",
      "SED ↓",
      "STDE ↑",
      "Vec ↑",
      "Dir ↑",
      "Pos ↑"
    ],
    "rows": [
      {
        "name": "GazeFormer",
        "ours": false,
        "values": [
          0.28,
          5.11,
          0.801,
          0.902,
          0.644,
          0.803
        ]
      },
      {
        "name": "HAT",
        "ours": false,
        "values": [
          0.296,
          5.07,
          0.8,
          0.909,
          0.649,
          0.825
        ]
      },
      {
        "name": "GazeSearch",
        "ours": false,
        "values": [
          0.351,
          4.89,
          0.809,
          0.917,
          0.679,
          0.829
        ]
      },
      {
        "name": "LogitGaze",
        "ours": false,
        "values": [
          0.328,
          5.07,
          0.81,
          0.882,
          0.643,
          0.809
        ]
      },
      {
        "name": "LogitGaze-Med (Res)",
        "ours": false,
        "values": [
          0.416,
          4.68,
          0.852,
          0.935,
          0.65,
          0.823
        ]
      },
      {
        "name": "LogitGaze-Med (CheX)",
        "ours": false,
        "values": [
          0.419,
          4.68,
          0.855,
          0.938,
          0.651,
          0.823
        ]
      },
      {
        "name": "Ours (MedCLIP)",
        "ours": true,
        "values": [
          0.479,
          3.74,
          0.909,
          0.941,
          0.649,
          0.837
        ]
      },
      {
        "name": "Ours (Res)",
        "ours": true,
        "values": [
          0.484,
          3.66,
          0.912,
          0.966,
          0.678,
          0.917
        ]
      },
      {
        "name": "Ours (CheX)",
        "ours": true,
        "values": [
          0.489,
          3.63,
          0.911,
          0.966,
          0.686,
          0.915
        ]
      }
    ]
  }
};
