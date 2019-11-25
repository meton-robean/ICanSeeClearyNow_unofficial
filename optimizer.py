import torch


# Optimizers
def Get_optimizers(args, generator, discriminator):
    optimizer_G = torch.optim.Adam(
                    generator.parameters(),
                    lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(
                    discriminator.parameters(),
                    lr=args.lr, betas=(args.b1, args.b2))

    return optimizer_G, optimizer_D



