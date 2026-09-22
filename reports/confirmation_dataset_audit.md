# Independent Holdout Dataset Audit Report

**Status**: **`HOLDOUT FROZEN`**  
**Creation Timestamp**: `2026-09-22T09:49:12+0800`  
**Primary Evaluation Corpus**: `D100`  
**Total Independent Questions**: `200`  

---

## 1. Executive Summary & Compliance Verification

| Requirement | Specification | Observed in Dataset | Compliance Status |
|:---|:---|:---|:---|
| **Independence from Old Benchmark** | Never seen in training/tuning ($N=200$) | Exactly 200 brand-new questions | **FULL COMPLIANCE** |
| **Old Gold Chunks Excluded** | Zero overlap with 180 old gold chunks | **0 overlap** (156 unique clean chunks) | **FULL COMPLIANCE** |
| **Max Jaccard vs Old Questions** | Strict upper limit $< 0.35$ | Max: **0.333**, Mean: **0.156** | **FULL COMPLIANCE** |
| **Banned Heuristic Topics** | 0 overlap with Q014, Q016, Q025, Q026, Q048, etc. | 0 banned topics matched | **FULL COMPLIANCE** |
| **Verbatim Span Grounding** | 100% exact substring in chunk `text` | 100% verified (`assert span in chunk['text']`) | **FULL COMPLIANCE** |
| **Hop Stratification** | ~40% 1-hop, ~35% 2-hop, ~25% 3-hop | **80 (40.0%) / 70 (35.0%) / 50 (25.0%)** | **FULL COMPLIANCE** |
| **Dataset Freeze** | Checksums locked prior to running systems | Manifest marked `HOLDOUT FROZEN` | **FULL COMPLIANCE** |

---

## 2. Hop Count & Structural Distribution

```text
Total Questions: 200
  ├── 1-hop (Single document factual / penalty / condition): 80 (40.0%)
  ├── 2-hop (Cross-document citation / harmonization / based-on): 70 (35.0%)
  └── 3-hop+ (Cross-document multi-statute chain / coordination): 50 (25.0%)
```

### Tag Frequency Breakdown
| Tag | Count | Description |
|:---|:---|:---|
| `single_hop` | 80 | Category tag |
| `2-hop` | 70 | Category tag |
| `3-hop` | 50 | Category tag |
| `based_on` | 41 | Category tag |
| `cross_statute` | 41 | Category tag |
| `procedure` | 32 | Category tag |
| `condition` | 24 | Category tag |
| `harmonization` | 24 | Category tag |
| `penalty` | 20 | Category tag |
| `multi_authority` | 7 | Category tag |
| `temporal` | 6 | Category tag |
| `exception` | 3 | Category tag |
| `definition` | 1 | Category tag |
| `reference` | 1 | Category tag |

---

## 3. Cryptographic Artifact Signatures (SHA-256)

| Artifact | File Path | SHA-256 Checksum |
|:---|:---|:---|
| **Confirmation Questions** | `benchmark/confirmation/questions.jsonl` | `d56d6a7f6647a452bc3145847f79a358878936478ebca98b016eedc3ab98fa66` |
| **Confirmation Gold Truth** | `benchmark/confirmation/gold.jsonl` | `ecf10815a4a23942ce96805f96c10e7a76b28cd11fe738545ef735bd798cd455` |
| **D100 Manifest** | `data/manifests/d100.json` | `852865dc239b65c10fc8e99b0469f43886b624f315ca6060eeb26a4666f0fd22` |
| **Chunks Database** | `data/chunks.jsonl` | `7763192631df6bd55a0db1188ae8fcf3e7ff212304a0a5e75605d4539efeca7a` |
| **LSDB SQLite Database** | `data/knowledge_lsdb.sqlite` | `5495e97f18f2e414622cca162f8f6051cf88b77e8d3aefd565dcb7950ec9bfa8` |

---

## 4. Sample Verification Entries

### [CONF_Q001] (1-hop, tags: ['single_hop', 'procedure'])
- **Question**: 根据《中华人民共和国食品安全法实施条例》，食品安全监督管理部门应当对企业食品安全管理人员进行何种考核？
- **Gold Answer**: 随机监督抽查考核
- **Gold Documents**: `['doc013']`
- **Gold Chunks**: `['doc013#c025']`
- **Gold Spans Count**: `1` (All 100% verified verbatim)
- **Max Jaccard vs Old Benchmark**: `0.304`

### [CONF_Q080] (1-hop, tags: ['single_hop', 'penalty'])
- **Question**: 根据《病原微生物实验室生物安全管理条例》，在不符合相应生物安全要求的实验室从事病原微生物相关实验活动的，由哪个部门责令停止有关活动并给予警告？
- **Gold Answer**: 由县级以上地方人民政府卫生主管部门、兽医主管部门依照各自职责责令停止有关活动，监督其将用于实验活动的病原微生物销毁或者送交保藏机构，并给予警告。
- **Gold Documents**: `['doc008']`
- **Gold Chunks**: `['doc008#c066']`
- **Gold Spans Count**: `1` (All 100% verified verbatim)
- **Max Jaccard vs Old Benchmark**: `0.182`

### [CONF_Q081] (2-hop, tags: ['2-hop', 'based_on'])
- **Question**: 根据《关于取得助产士（师）资格人员不能认定执业医师资格的批复》及其关联规定，取得助产士（师）资格的人员能否被认定为执业医师？若此类人员未经执业注册私自开展家庭接生造成人员死亡，应如何适用法律和处理？
- **Gold Answer**: 取得助产士（师）资格的人员不能认定执业医师资格。对于取得医师资格但未经注册取得医师执业证书而从事医师执业活动的人员，按照《中华人民共和国执业医师法》第三十九条的规定处理，造成患者人身损害的，按照《医疗事故处理条例》第六十一条的规定处理。
- **Gold Documents**: `['doc048', 'doc100']`
- **Gold Chunks**: `['doc048#c002', 'doc100#c001']`
- **Gold Spans Count**: `2` (All 100% verified verbatim)
- **Max Jaccard vs Old Benchmark**: `0.294`

### [CONF_Q150] (2-hop, tags: ['2-hop', 'harmonization'])
- **Question**: 根据卫生部《关于产妇分娩后胎盘处理问题的批复》，产妇分娩后胎盘应当归谁所有？在什么情况下医疗机构可以处置胎盘？如果胎盘可能造成传染病传播，医疗机构应当如何处理？该处理方式与《医疗废物管理条例》有何衔接关系？
- **Gold Answer**: 根据《关于产妇分娩后胎盘处理问题的批复》，产妇分娩后胎盘应当归产妇所有。产妇放弃或者捐献胎盘的，可以由医疗机构进行处置。任何单位和个人不得买卖胎盘。如果胎盘可能造成传染病传播的，医疗机构应当及时告知产妇，按照《传染病防治法》、《医疗废物管理条例》的有关规定进行消毒处理，并按照医疗废物进行处置。这意味着，当胎盘可能造成传染病传播时，其处理需遵循《医疗废物管理条例》的相关规定，即按照医疗废物进行管理和处置，包括消毒处理等要求。
- **Gold Documents**: `['doc083', 'doc019']`
- **Gold Chunks**: `['doc083#c001', 'doc019#c001']`
- **Gold Spans Count**: `2` (All 100% verified verbatim)
- **Max Jaccard vs Old Benchmark**: `0.158`

### [CONF_Q151] (3-hop, tags: ['3-hop', 'cross_statute'])
- **Question**: 某市卫生局在审查一家新设立的社区卫生服务中心的医疗广告时，需要依据《卫生部关于医疗广告审查中有关问题的批复》判断该机构是否属于《医疗机构管理条例实施细则》第三条规定的医疗机构类别，同时须遵守《医疗机构管理条例》第五条关于监督管理职责的规定。请问：该社区卫生服务中心是否属于合法医疗机构类别？其医疗广告审查应遵循何种程序？并说明市卫生局在审查中应如何履行监督管理职责？
- **Gold Answer**: 根据《卫生部关于医疗广告审查中有关问题的批复》（卫医函〔2008〕25号），医疗广告审查中有关医疗机构类别的认定，应当依据《医疗机构管理条例实施细则》的规定。而《医疗机构管理条例实施细则》第三条经修订后，明确将“社区卫生服务中心、社区卫生服务站”列为医疗机构类别之一（第三项）。因此，该社区卫生服务中心属于合法医疗机构类别。在医疗广告审查程序上，该批复要求依据《医疗机构管理条例实施细则》进行审查，故市卫生局应按照该细则规定的类别标准审核其广告内容。同时，根据《医疗机构管理条例》第五条，县级以上地方人民政府卫生行政部门负责本行政区域内医疗机构的监督管理工作，因此该市卫生局作为县级以上地方卫生行政部门，有权并应当对该社区卫生服务中心的医疗广告进行审查和监督管理，确保其符合法规要求。
- **Gold Documents**: `['doc058', 'doc010', 'doc033']`
- **Gold Chunks**: `['doc058#c001', 'doc010#c005', 'doc033#c007']`
- **Gold Spans Count**: `3` (All 100% verified verbatim)
- **Max Jaccard vs Old Benchmark**: `0.149`

### [CONF_Q200] (3-hop, tags: ['3-hop', 'cross_statute'])
- **Question**: 某心内科医师已在设有超声心动图检查室的心内科执业，注册执业范围为内科专业并从事心血管疾病诊疗工作，但尚未注册取得医师执业证书。若其独立为患者进行超声心动图检查并出具诊断报告，后因操作失误造成患者人身损害，应如何适用法律？请结合《关于心内科医师从事超声心动图检查有关问题的批复》《卫生部关于未经执业注册医师私自开展家庭接生造成人员死亡有关法律适用和案件移送问题的批复》和《医疗事故处理条例》的相关规定，说明该医师的执业资格认定、行政法律责任及民事赔偿处理依据。
- **Gold Answer**: 该医师虽符合《关于心内科医师从事超声心动图检查有关问题的批复》（卫医政函〔2010〕91号）规定的两项条件（在设有超声心动图检查室的心内科中执业；注册执业范围为内科专业，并从事心血管疾病诊疗工作），但该批复的适用前提是“心内科执业医师”，即已注册取得医师执业证书的医师。该医师尚未注册取得医师执业证书，属于“取得医师资格但未经注册取得医师执业证书而从事医师执业活动的人员”，不能以该批复主张合法执业。根据《卫生部关于未经执业注册医师私自开展家庭接生造成人员死亡有关法律适用和案件移送问题的批复》（卫政法发〔2006〕483号），对于取得医师资格但未经注册取得医师执业证书而从事医师执业活动的人员，按照《中华人民共和国执业医师法》第三十九条的规定处理；造成患者人身损害的，按照《医疗事故处理条例》第六十一条的规定处理。因此，该医师的超声心动图检查及出具诊断报告行为属于未经注册非法从事医师执业活动，应依《执业医师法》第三十九条予以取缔、没收违法所得及罚款等行政处罚；因其造成患者人身损害，还应依《医疗事故处理条例》第六十一条处理，即非法行医造成患者人身损害，不属于医疗事故，触犯刑律的依法追究刑事责任，有关赔偿由受害人直接向人民法院提起诉讼。
- **Gold Documents**: `['doc080', 'doc100', 'doc015']`
- **Gold Chunks**: `['doc080#c001', 'doc100#c001', 'doc015#c001']`
- **Gold Spans Count**: `3` (All 100% verified verbatim)
- **Max Jaccard vs Old Benchmark**: `0.145`

