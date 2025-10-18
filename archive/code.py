# import cProfile, pstats

# def run_profiled():
#     run_rtrl(tt)  # call your entry function

# with cProfile.Profile() as pr:
#     run_profiled()

# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(30)

# loss = torch.zeros(1, device=device).requires_grad_()
# for t in range(x_chunk.size(0)):
#     xt = x_chunk[t][None, None, ...]        # [1, 1, D]
#     yt = y_chunk[t].unsqueeze(0)            # [1]
#     opt.zero_grad()
#     loss_t = criterion(logits, yt).mean()  # scalar
#     loss = loss + loss_t

#     # Only backprop if target is valid; always advance state
#     if yt.item() != -100:
#         loss = loss + loss_t
#         running = (0.98 * running + 0.02 * loss_t.item()) if running else loss_t.item()
#     h = h_new.detach().requires_grad_()
#     writer.add_scalar("train/loss", loss.item(), epoch*SEQ_LEN+start+t)
#     if t%10==0: print(f"loss: {running:.4f}")