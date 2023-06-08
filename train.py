import torch

device = torch.device("cuda")

def train(dataloader, model, optimizer_list, log):

    model.train()

    for batch, (v, x, y) in enumerate(dataloader):
        v, x, y = v.squeeze().to(device), x.squeeze().to(device), y.squeeze().to(device)

        # training step by step: set retain_graph=False and step() after every loss

        optimizer_list['minet'].zero_grad()
        lld_loss = model.get_lld_loss(v, x, y)
        lld_loss.backward()
        optimizer_list['minet'].step()

        optimizer_list['c_rep'].zero_grad()
        optimizer_list['z_rep'].zero_grad()
        mi_loss = model.get_mi_loss(v, x, y)
        mi_loss.backward()
        optimizer_list['c_rep'].step()
        optimizer_list['z_rep'].step()

        optimizer_list['c_rep'].zero_grad()
        optimizer_list['z_rep'].zero_grad()
        optimizer_list['reg_x'].zero_grad()
        x_loss = model.get_x_loss(v, x) #+ model.get_reg_penalty(model.regressionX)
        x_loss.backward()
        optimizer_list['c_rep'].step()
        optimizer_list['z_rep'].step()
        optimizer_list['reg_x'].step()

        optimizer_list['c_rep'].zero_grad()
        optimizer_list['z_rep'].zero_grad()
        optimizer_list['reg_x'].zero_grad()
        optimizer_list['reg_y'].zero_grad()
        y_loss = model.get_y_loss(v, y) #+ model.get_reg_penalty(model.regressionY)
        y_loss.backward()
        optimizer_list['c_rep'].step()
        optimizer_list['z_rep'].step()
        optimizer_list['reg_x'].step()
        optimizer_list['reg_y'].step()

        if batch % 50 == 0:
            log.write("\nBatch {}".format(batch))
            log.write("lld_loss = %.6f" % lld_loss.item())
            log.write("mi_loss = %.6f" % mi_loss.item())
            log.write("x_loss = %.6f" % x_loss.item())
            log.write("y_loss = %.6f" % y_loss.item())

        


def test(dataloader, model, log):

    model.eval()

    lld_loss, mi_loss, x_loss, y_loss, y_mape = 0, 0, 0, 0, 0

    with torch.no_grad():
        for batch, (v, x, y) in enumerate(dataloader):
            v, x, y = v.squeeze().to(device), x.squeeze().to(device), y.squeeze().to(device)

            lld_loss += model.get_lld_loss(v, x, y).item()
            mi_loss += model.get_mi_loss(v, x, y).item()
            x_loss += model.get_x_loss(v, x).item()
            y_loss += model.get_y_loss(v, y).item()
            y_mape += model.get_mape(v, y).item()

    log.write("\nTest Error")
    log.write("lld_loss = %.6f" % (lld_loss / len(dataloader)))
    log.write("mi_loss = %.6f" % (mi_loss / len(dataloader)))
    log.write("x_loss = %.6f" % (x_loss / len(dataloader)))
    log.write("y_loss = %.6f" % (y_loss / len(dataloader)))
    log.write("y_mape = %.6f" % (y_mape / len(dataloader)))


