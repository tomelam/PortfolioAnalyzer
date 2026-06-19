| Phase | Purpose | Bad to use DataFrame? | Reason |
| --- | --- | --- | --- |
| Building assets | Turn each fund into an AssetTimeseries | ❌ Bad | Each asset expects just one Series |
| Calculating returns | Separate daily returns, weight separately | ❌ Bad | Different assets may need custom logic |
| Assigning custom weights | Portfolio construction | ❌ Bad | Must treat assets individually |
| Calculating metrics | Annualized return, volatility per fund | ❌ Bad | Easier when each is an object (not a column) |
| Saving golden data | Save "true" aligned series separately | ❌ Bad | Want fine-grained control |
