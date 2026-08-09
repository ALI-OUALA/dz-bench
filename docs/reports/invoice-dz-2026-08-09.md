# Invoice-DZ baseline report — 2026-08-09

This checkpoint uses four original synthetic Algerian invoices generated with seed 23:
clean, compressed, photographed, and skewed. It is a pipeline regression suite, not a
claim about real invoice accuracy. All four pages scored; none were omitted or retried.

| System | CER | WER | Reading order | Field F1 | Financial | Runtime/page | RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DzDoc PP-OCRv5 Arabic only | 0.6475 | 0.7857 | 0.2800 | 0.8829 | 0.9000 | 52.26 s | 365.7 MB |
| DzDoc PP-OCRv5 routed | 0.6516 | 0.7812 | 0.2900 | 0.9050 | 0.9500 | 40.47 s | 371.5 MB |

Routed OCR was 22.5% faster and improved structured extraction, though it did not beat
Arabic-only recognition on CER. Both had zero digit-exact page accuracy and zero table
structure similarity. The routed pack reached 0.9853 field precision, 0.8375 recall,
0.8375 exact accuracy, and zero structured hallucinations on this authored slice. Next
work remains semantic table reconstruction and better raw mixed-script OCR.

PaddleOCR-VL-1.6 was selected as the guarded fallback candidate. On the clean page with
one escalation maximum, it improved CER from 0.6831 to 0.6093 while field F1 remained
0.9474 and financial accuracy remained 1.0. Runtime increased from 15.67 to 87.89
seconds; the accepted VLM event took 71.73 seconds. This is useful quality evidence for
hard high-value regions and equally clear evidence against making VLM the default.

Raw reports were written outside the repository under
`E:\dev\data\dzdoc-invoice-bench`. They can be regenerated from the committed generator
and public contracts; model assets remain external.
