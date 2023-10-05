#include "taco/lower/mode_format_vb.h"

#include "ir/ir_generators.h"
#include "taco/ir/simplify.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

    VariableBlockModeFormat::VariableBlockModeFormat() :
            VariableBlockModeFormat(false, true, true, false, false) {
    }

    VariableBlockModeFormat::VariableBlockModeFormat(bool isFull, bool isOrdered,
                                               bool isUnique, bool isZeroless,
                                               bool isLastValueFill,
                                               long long allocSize) :
            ModeFormatImpl("variableblock", isFull, isOrdered, isUnique, false, true,
                           isZeroless, true, false, false, false,
                           true, false, false,
                           false, false, false,
                           taco_positer_kind::NONE,
                           isLastValueFill),
            allocSize(allocSize) {
    }

    ModeFormat VariableBlockModeFormat::copy(
            vector<ModeFormat::Property> properties) const {
      bool isFull = this->isFull;
      bool isOrdered = this->isOrdered;
      bool isUnique = this->isUnique;
      bool isZeroless = this->isZeroless;
      for (const auto property : properties) {
        switch (property) {
          case ModeFormat::FULL:
            isFull = true;
            break;
          case ModeFormat::NOT_FULL:
            isFull = false;
            break;
          case ModeFormat::ORDERED:
            isOrdered = true;
            break;
          case ModeFormat::NOT_ORDERED:
            isOrdered = false;
            break;
          case ModeFormat::UNIQUE:
            isUnique = true;
            break;
          case ModeFormat::NOT_UNIQUE:
            isUnique = false;
            break;
          case ModeFormat::ZEROLESS:
            isZeroless = true;
            break;
          case ModeFormat::NOT_ZEROLESS:
            isZeroless = false;
            break;
          default:
            break;
        }
      }
      const auto compressedVariant =
              std::make_shared<VariableBlockModeFormat>(isFull, isOrdered, isUnique,
                                                     isZeroless);
      return ModeFormat(compressedVariant);
    }

    ModeFunction VariableBlockModeFormat::coordIterBounds(std::vector<ir::Expr> parentCoords, Mode mode) const {
      auto end = GetProperty::make(mode.getTensorExpr(), TensorProperty::Dimension);
      return ModeFunction(Stmt(), {0, end});

//      Expr pbegin = Load::make(getPosArray(mode.getModePack()), parentPos);
//      Expr pend = Load::make(getPosArray(mode.getModePack()),
//                             ir::Add::make(parentPos, 1));
//      return ModeFunction(Stmt(), {pbegin, pend});
    }

    ModeFunction VariableBlockModeFormat::coordIterAccess(ir::Expr parentPos, std::vector<ir::Expr> coords,
                                                          Mode mode) const {
      return ModeFunction();
    }

    ModeFunction VariableBlockModeFormat::posIterBounds(Expr parentPos,
                                                     Mode mode) const {
      Expr pbegin = Load::make(getPosArray(mode.getModePack()), parentPos);
      Expr pend = Load::make(getPosArray(mode.getModePack()),
                             ir::Add::make(parentPos, 1));
      return ModeFunction(Stmt(), {pbegin, pend});
    }

    ModeFunction VariableBlockModeFormat::coordBounds(Expr parentPos,
                                                   Mode mode) const {
      taco_not_supported_yet;
      return ModeFunction();

//      Expr pend = Load::make(getPosArray(mode.getModePack()),
//                             ir::Add::make(parentPos, 1));
//      Expr coordend = Load::make(getCoordArray(mode.getModePack()), ir::Sub::make(pend, 1));
//      return ModeFunction(Stmt(), {0, coordend});
    }

    ModeFunction VariableBlockModeFormat::posIterAccess(std::vector<ir::Expr> pos_,
                                                     std::vector<ir::Expr> coords,
                                                     ir::Expr values, Datatype type,
                                                     Mode mode) const {
      taco_iassert(mode.getPackLocation() == 0);
      taco_iassert(pos_.size() > 1);
      ir::Expr pos = pos_.back();

//      std::cout << "positions: ";
//      for(auto& p : pos_){
//        std::cout << p << ", ";
//      }
//      std::cout << std::endl;
//
//      std::cout << "coords: ";
//      for(auto& c : coords){
//        std::cout << c << ", ";
//      }
//      std::cout << std::endl;

      Expr idxArray = getCoordArray(mode.getModePack());
      Expr stride = (int)mode.getModePack().getNumModes();
      Expr idx = Load::make(idxArray, ir::Mul::make(pos_[pos_.size()-2], stride));

      Expr pbegin = 0;
      if (pos_.size() > 1)
        pbegin = Load::make(getPosArray(mode.getModePack()),pos_[pos_.size()-2]);

      Expr blk_coord = ir::Sub::make(pos, pbegin);
      return ModeFunction(Stmt(), {ir::Add::make(idx, blk_coord), true});
    }

    vector<Expr> VariableBlockModeFormat::getArrays(Expr tensor, int mode,
                                                 int level) const {
      std::string arraysName = util::toString(tensor) + std::to_string(level);
      return {GetProperty::make(tensor, TensorProperty::Indices,
                                level - 1, 0, arraysName + "_pos"),
              GetProperty::make(tensor, TensorProperty::Indices,
                                level - 1, 1, arraysName + "_crd")};
    }

    Expr VariableBlockModeFormat::getPosArray(ModePack pack) const {
      return pack.getArray(0);
    }

    Expr VariableBlockModeFormat::getCoordArray(ModePack pack) const {
      return pack.getArray(1);
    }

    Expr VariableBlockModeFormat::getPosCapacity(Mode mode) const {
      const std::string varName = mode.getName() + "_pos_size";

      if (!mode.hasVar(varName)) {
        Expr posCapacity = Var::make(varName, Int());
        mode.addVar(varName, posCapacity);
        return posCapacity;
      }

      return mode.getVar(varName);
    }

    Expr VariableBlockModeFormat::getCoordCapacity(Mode mode) const {
      const std::string varName = mode.getName() + "_crd_size";

      if (!mode.hasVar(varName)) {
        Expr idxCapacity = Var::make(varName, Int());
        mode.addVar(varName, idxCapacity);
        return idxCapacity;
      }

      return mode.getVar(varName);
    }


    Expr VariableBlockModeFormat::getWidth(Mode mode) const {
      return ir::Literal::make(allocSize, Datatype::Int32);
    }

    bool VariableBlockModeFormat::equals(const ModeFormatImpl& other) const {
      return ModeFormatImpl::equals(other) &&
             (dynamic_cast<const VariableBlockModeFormat&>(other).allocSize == allocSize);
    }

}
