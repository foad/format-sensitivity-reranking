# demo01108: Into which body of water does the Zelish River flow?

Expected answer: the Gulf of Durish. Wrong answer on offer: the Athwen Sea.

Prompt: Answer the question using only the passages below. If they do not contain the answer, say so.

## untrained (base), mixed

Gold passage lost at rank 4 of 10.

```text
Question: Into which body of water does the Zelish River flow?

Passage 1:
**name**: Zelish River
**length**: 328 km
**source**: the Holtpan Hills
**basinCountries**: Ostpania, Relephia, Kesquoland
**tributaries**: the Calcal and the Wenmal
**discharge**: 8,745 m3/s

The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. Its mean discharge is about 8,745 m3/s. Its main tributaries are the Calcal and the Wenmal. A number of dams regulate its flow. Its valley is known for seasonal flooding.

Passage 2:
**name**: Zelish River
**source**: the Holtpan Hills
**length**: 328 km

The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. Its mean discharge is about 8,745 m3/s. Its main tributaries are the Calcal and the Wenmal. The river is 328 km long. It rises at the Holtpan Hills. Several towns along its banks depend on it for irrigation.

Passage 3:
name=Zelish River length=328 km discharge=8,745 m3/s source=the Holtpan Hills tributaries=the Calcal and the Wenmal
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. It rises at the Holtpan Hills. Its mean discharge is about 8,745 m3/s. The river is 328 km long. Its main tributaries are the Calcal and the Wenmal. A number of dams regulate its flow.
```

## untrained (base), standardised to yaml

Gold passage lost at rank 4 of 10.

```text
Question: Into which body of water does the Zelish River flow?

Passage 1:
---
name: Zelish River
length: 328 km
source: the Holtpan Hills
basinCountries: Ostpania, Relephia, Kesquoland
tributaries: the Calcal and the Wenmal
discharge: 8,745 m3/s
---
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. Its mean discharge is about 8,745 m3/s. Its main tributaries are the Calcal and the Wenmal. A number of dams regulate its flow. Its valley is known for seasonal flooding.

Passage 2:
---
name: Zelish River
length: 328 km
discharge: 8,745 m3/s
source: the Holtpan Hills
tributaries: the Calcal and the Wenmal
---
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. It rises at the Holtpan Hills. Its mean discharge is about 8,745 m3/s. The river is 328 km long. Its main tributaries are the Calcal and the Wenmal. A number of dams regulate its flow.

Passage 3:
---
name: Zelish River
source: the Holtpan Hills
length: 328 km
---
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. Its mean discharge is about 8,745 m3/s. Its main tributaries are the Calcal and the Wenmal. The river is 328 km long. It rises at the Holtpan Hills. Several towns along its banks depend on it for irrigation.
```

## rank only (5fmt_lam0), mixed

Gold passage lost at rank 4 of 10.

```text
Question: Into which body of water does the Zelish River flow?

Passage 1:
**name**: Zelish River
**length**: 328 km
**source**: the Holtpan Hills
**basinCountries**: Ostpania, Relephia, Kesquoland
**tributaries**: the Calcal and the Wenmal
**discharge**: 8,745 m3/s

The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. Its mean discharge is about 8,745 m3/s. Its main tributaries are the Calcal and the Wenmal. A number of dams regulate its flow. Its valley is known for seasonal flooding.

Passage 2:
**name**: Zelish River
**source**: the Holtpan Hills
**length**: 328 km

The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. Its mean discharge is about 8,745 m3/s. Its main tributaries are the Calcal and the Wenmal. The river is 328 km long. It rises at the Holtpan Hills. Several towns along its banks depend on it for irrigation.

Passage 3:
name=Zelish River length=328 km discharge=8,745 m3/s source=the Holtpan Hills tributaries=the Calcal and the Wenmal
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. It rises at the Holtpan Hills. Its mean discharge is about 8,745 m3/s. The river is 328 km long. Its main tributaries are the Calcal and the Wenmal. A number of dams regulate its flow.
```

## rank only (5fmt_lam0), standardised to yaml

Gold passage lost at rank 4 of 10.

```text
Question: Into which body of water does the Zelish River flow?

Passage 1:
---
name: Zelish River
length: 328 km
source: the Holtpan Hills
basinCountries: Ostpania, Relephia, Kesquoland
tributaries: the Calcal and the Wenmal
discharge: 8,745 m3/s
---
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. Its mean discharge is about 8,745 m3/s. Its main tributaries are the Calcal and the Wenmal. A number of dams regulate its flow. Its valley is known for seasonal flooding.

Passage 2:
---
name: Zelish River
source: the Holtpan Hills
length: 328 km
---
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. Its mean discharge is about 8,745 m3/s. Its main tributaries are the Calcal and the Wenmal. The river is 328 km long. It rises at the Holtpan Hills. Several towns along its banks depend on it for irrigation.

Passage 3:
---
name: Zelish River
length: 328 km
discharge: 8,745 m3/s
source: the Holtpan Hills
tributaries: the Calcal and the Wenmal
---
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. It rises at the Holtpan Hills. Its mean discharge is about 8,745 m3/s. The river is 328 km long. Its main tributaries are the Calcal and the Wenmal. A number of dams regulate its flow.
```

## invariance (5fmt_lam1), mixed

Gold passage kept at rank 1 of 10.

```text
Question: Into which body of water does the Zelish River flow?

Passage 1:
{"name": "Zelish River", "discharge": "8,745 m3/s", "length": "328 km", "source": "the Holtpan Hills", "mouth": "the Gulf of Durish", "basinCountries": "Ostpania, Relephia, Kesquoland"}
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. The river is 328 km long. It rises at the Holtpan Hills. A number of dams regulate its flow. Several towns along its banks depend on it for irrigation.

Passage 2:
**name**: Zelish River
**length**: 328 km
**source**: the Holtpan Hills
**basinCountries**: Ostpania, Relephia, Kesquoland
**tributaries**: the Calcal and the Wenmal
**discharge**: 8,745 m3/s

The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. Its mean discharge is about 8,745 m3/s. Its main tributaries are the Calcal and the Wenmal. A number of dams regulate its flow. Its valley is known for seasonal flooding.

Passage 3:
name=Zelish River length=328 km discharge=8,745 m3/s source=the Holtpan Hills tributaries=the Calcal and the Wenmal
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. It rises at the Holtpan Hills. Its mean discharge is about 8,745 m3/s. The river is 328 km long. Its main tributaries are the Calcal and the Wenmal. A number of dams regulate its flow.
```

## invariance (5fmt_lam1), standardised to yaml

Gold passage kept at rank 3 of 10.

```text
Question: Into which body of water does the Zelish River flow?

Passage 1:
---
name: Zelish River
length: 328 km
source: the Holtpan Hills
basinCountries: Ostpania, Relephia, Kesquoland
tributaries: the Calcal and the Wenmal
discharge: 8,745 m3/s
---
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. Its mean discharge is about 8,745 m3/s. Its main tributaries are the Calcal and the Wenmal. A number of dams regulate its flow. Its valley is known for seasonal flooding.

Passage 2:
---
name: Zelish River
source: the Holtpan Hills
length: 328 km
---
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. Its mean discharge is about 8,745 m3/s. Its main tributaries are the Calcal and the Wenmal. The river is 328 km long. It rises at the Holtpan Hills. Several towns along its banks depend on it for irrigation.

Passage 3:
---
name: Zelish River
discharge: 8,745 m3/s
length: 328 km
source: the Holtpan Hills
mouth: the Gulf of Durish
basinCountries: Ostpania, Relephia, Kesquoland
---
The Zelish River is a river that flows through Ostpania, Relephia, Kesquoland. The river is 328 km long. It rises at the Holtpan Hills. A number of dams regulate its flow. Several towns along its banks depend on it for irrigation.
```
